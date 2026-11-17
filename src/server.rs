//! Solr-compatible HTTP API: the subset of `/select` and `/update` that the
//! existing clients use, with Solr's request parameters and response shapes.
//!
//! Every update request is committed (and visible) when it returns, which is
//! stricter than Solr's default of waiting for an explicit or auto commit.

use crate::index::{Batch, Index, IndexError, SearchRequest, SortField};
use crate::query::{ParseOptions, Query, parse};
use axum::body::Bytes;
use axum::extract::{DefaultBodyLimit, Path, RawQuery, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Map, Value, json};
use std::sync::Arc;
use std::time::Instant;

/// Bulk updates are routinely tens of megabytes; Solr's own default limit is
/// about 2 GB. 512 MB keeps a single request from exhausting memory.
const MAX_UPDATE_BYTES: usize = 512 * 1024 * 1024;

pub struct AppState {
    pub core: String,
    pub index: Index,
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/solr/{core}/select", get(select))
        .route("/solr/{core}/update", post(update))
        .route("/solr/{core}/update/json/docs", post(update))
        .layer(DefaultBodyLimit::max(MAX_UPDATE_BYTES))
        .with_state(state)
}

struct SolrError(StatusCode, String);

impl IntoResponse for SolrError {
    fn into_response(self) -> Response {
        let code = self.0.as_u16();
        let body = json!({
            "responseHeader": {"status": code, "QTime": 0},
            "error": {"msg": self.1, "code": code},
        });
        (self.0, Json(body)).into_response()
    }
}

fn bad_request(msg: impl ToString) -> SolrError {
    SolrError(StatusCode::BAD_REQUEST, msg.to_string())
}

impl From<IndexError> for SolrError {
    fn from(e: IndexError) -> Self {
        match e {
            IndexError::Doc(d) => bad_request(d),
            other => SolrError(StatusCode::INTERNAL_SERVER_ERROR, other.to_string()),
        }
    }
}

fn check_core(state: &AppState, core: &str) -> Result<(), SolrError> {
    if core == state.core {
        Ok(())
    } else {
        Err(SolrError(
            StatusCode::NOT_FOUND,
            format!("no such core: {core}"),
        ))
    }
}

struct Params(Vec<(String, String)>);

impl Params {
    fn parse(raw: Option<&str>) -> Self {
        Params(
            form_urlencoded::parse(raw.unwrap_or_default().as_bytes())
                .into_owned()
                .collect(),
        )
    }

    fn get(&self, key: &str) -> Option<&str> {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    fn all(&self, key: &str) -> Vec<&str> {
        self.0
            .iter()
            .filter(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    fn usize(&self, key: &str, default: usize) -> Result<usize, SolrError> {
        self.get(key).map_or(Ok(default), |v| {
            v.trim()
                .parse()
                .map_err(|_| bad_request(format!("invalid {key}: {v}")))
        })
    }

    /// Solr's `echoParams=explicit`: repeated keys become arrays.
    fn echo(&self) -> Value {
        let mut out = Map::new();
        for (k, v) in &self.0 {
            match out.get_mut(k) {
                None => {
                    out.insert(k.clone(), Value::String(v.clone()));
                }
                Some(Value::Array(a)) => a.push(Value::String(v.clone())),
                Some(prev) => *prev = Value::Array(vec![prev.clone(), Value::String(v.clone())]),
            }
        }
        Value::Object(out)
    }
}

fn parse_sort(spec: &str) -> Result<Vec<SortField>, SolrError> {
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|clause| {
            let mut parts = clause.split_whitespace();
            let field = parts.next().unwrap_or_default().to_string();
            let descending = match parts.next().map(str::to_ascii_lowercase).as_deref() {
                Some("asc") => false,
                Some("desc") => true,
                _ => {
                    return Err(bad_request(format!(
                        "can't determine sort order for '{clause}', expected asc or desc"
                    )));
                }
            };
            Ok(SortField { field, descending })
        })
        .collect()
}

async fn select(
    State(state): State<Arc<AppState>>,
    Path(core): Path<String>,
    RawQuery(raw): RawQuery,
) -> Result<Json<Value>, SolrError> {
    let started = Instant::now();
    check_core(&state, &core)?;
    let params = Params::parse(raw.as_deref());
    let schema = state.index.schema();
    let opts = ParseOptions {
        default_field: params.get("df").unwrap_or("_text_").to_string(),
        default_and: params
            .get("q.op")
            .is_some_and(|op| op.eq_ignore_ascii_case("AND")),
    };
    let query = match params.get("q").map(str::trim) {
        None | Some("") => Query::MatchNone,
        Some(q) => parse(q, schema, &opts).map_err(bad_request)?,
    };
    let filters = params
        .all("fq")
        .into_iter()
        .filter(|f| !f.trim().is_empty())
        .map(|f| parse(f, schema, &opts).map_err(bad_request))
        .collect::<Result<Vec<_>, _>>()?;
    let sort = parse_sort(params.get("sort").unwrap_or_default())?;
    let start = params.usize("start", 0)?;
    let rows = params.usize("rows", 10)?;

    let fl: Vec<&str> = params
        .get("fl")
        .unwrap_or("*")
        .split([',', ' '])
        .map(str::trim)
        .filter(|f| !f.is_empty())
        .collect();
    let stored_fl: Vec<&str> = fl.iter().copied().filter(|f| *f != "score").collect();
    let searcher = state.index.cached_searcher().await?;
    let request = SearchRequest {
        query,
        filters,
        sort,
        start,
        rows,
    };
    let wanted = (!fl.is_empty() && !fl.contains(&"*")).then_some(stored_fl.as_slice());
    let result = searcher
        .search_fields(&request, wanted)
        .map_err(bad_request)?;

    let all_fields = fl.is_empty() || fl.contains(&"*");
    let with_score = fl.contains(&"score");
    let docs: Vec<Value> = result
        .docs
        .into_iter()
        .map(|hit| {
            let mut doc: Map<String, Value> = hit
                .doc
                .into_iter()
                .filter(|(k, _)| all_fields || fl.contains(&k.as_str()))
                .collect();
            if with_score {
                doc.insert("score".into(), json!(hit.score));
            }
            Value::Object(doc)
        })
        .collect();

    let mut response =
        json!({"numFound": result.num_found, "start": start, "numFoundExact": true, "docs": docs});
    if with_score {
        response["maxScore"] = json!(result.max_score);
    }
    Ok(Json(json!({
        "responseHeader": {"status": 0, "QTime": started.elapsed().as_millis() as u64, "params": params.echo()},
        "response": response,
    })))
}

fn add_delete(batch: &mut Batch, v: &Value) -> Result<(), SolrError> {
    match v {
        Value::String(id) => batch.push_delete(id),
        Value::Number(n) => batch.push_delete(&n.to_string()),
        Value::Array(items) => items.iter().try_for_each(|i| add_delete(batch, i))?,
        Value::Object(o) if o.contains_key("id") => add_delete(batch, &o["id"])?,
        Value::Object(o) if o.contains_key("query") => {
            return Err(bad_request("delete by query is not supported yet"));
        }
        other => return Err(bad_request(format!("invalid delete command: {other}"))),
    }
    Ok(())
}

fn add_add(batch: &mut Batch, v: &Value) -> Result<(), SolrError> {
    match v {
        Value::Array(items) => items.iter().try_for_each(|i| add_add(batch, i)),
        Value::Object(o) => match o.get("doc") {
            Some(doc @ Value::Object(_)) => {
                batch.push_add(doc.clone());
                Ok(())
            }
            _ => Err(bad_request("add command requires a 'doc' object")),
        },
        other => Err(bad_request(format!("invalid add command: {other}"))),
    }
}

const COMMANDS: &[&str] = &["add", "delete", "commit", "optimize"];

async fn update(
    State(state): State<Arc<AppState>>,
    Path(core): Path<String>,
    body: Bytes,
) -> Result<Json<Value>, SolrError> {
    let started = Instant::now();
    check_core(&state, &core)?;
    let body: Value = if body.is_empty() {
        Value::Null
    } else {
        serde_json::from_slice(&body).map_err(bad_request)?
    };
    let mut batch = Batch::default();
    match &body {
        Value::Null => {}
        Value::Array(docs) => {
            for d in docs {
                if !d.is_object() {
                    return Err(bad_request(format!("expected a document object, got {d}")));
                }
                batch.push_add(d.clone());
            }
        }
        Value::Object(o) if o.keys().any(|k| COMMANDS.contains(&k.as_str())) => {
            // Commands apply in document order (serde_json preserves it).
            for (cmd, v) in o {
                match cmd.as_str() {
                    "add" => add_add(&mut batch, v)?,
                    "delete" => add_delete(&mut batch, v)?,
                    "commit" | "optimize" => {}
                    other => return Err(bad_request(format!("unknown update command '{other}'"))),
                }
            }
        }
        Value::Object(_) => batch.push_add(body.clone()),
        other => return Err(bad_request(format!("unexpected update body: {other}"))),
    }
    if !batch.is_empty() {
        state.index.commit(batch).await?;
        // Read-your-writes on this instance; others see it on their next refresh.
        state.index.searcher().await?;
    }
    Ok(Json(
        json!({"responseHeader": {"status": 0, "QTime": started.elapsed().as_millis() as u64}}),
    ))
}
