//! Copies every document out of an existing Solr core, using cursorMark
//! paging, and commits each page as one segment.

use crate::index::{Batch, Index, IndexError};
use crate::schema::Schema;
use serde_json::{Map, Value};

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("request to Solr failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("unexpected Solr response: {0}")]
    Response(String),
    #[error(transparent)]
    Index(#[from] IndexError),
}

/// Drops fields this engine recomputes or Solr manages itself: copy-field
/// destinations (rebuilt from their sources at index time) and `_version_`.
pub fn clean_doc(schema: &Schema, doc: Map<String, Value>) -> Map<String, Value> {
    doc.into_iter()
        .filter(|(k, _)| k != "_version_" && !schema.is_copy_destination(k))
        .collect()
}

pub struct ImportOptions {
    /// e.g. `https://host/solr/collection`
    pub core_url: String,
    /// Value for the `Authorization` header, e.g. `Basic <base64>`.
    pub authorization: Option<String>,
    pub page_size: usize,
    pub limit: Option<usize>,
}

pub async fn import_from_solr(
    index: &Index,
    opts: &ImportOptions,
    mut progress: impl FnMut(usize, u64),
) -> Result<usize, ImportError> {
    let client = reqwest::Client::new();
    let key = &index.schema().unique_key;
    let mut cursor = "*".to_string();
    let mut imported = 0usize;
    loop {
        let rows = opts
            .limit
            .map_or(opts.page_size, |l| opts.page_size.min(l - imported));
        if rows == 0 {
            break;
        }
        let rows_s = rows.to_string();
        let sort = format!("{key} asc");
        let url = format!("{}/select", opts.core_url.trim_end_matches('/'));
        let mut req = client.get(&url).query(&[
            ("q", "*:*"),
            ("sort", sort.as_str()),
            ("rows", rows_s.as_str()),
            ("cursorMark", cursor.as_str()),
            ("wt", "json"),
        ]);
        if let Some(auth) = &opts.authorization {
            req = req.header("Authorization", auth);
        }
        let res = req.send().await?;
        let status = res.status();
        if !status.is_success() {
            return Err(ImportError::Response(format!("HTTP {status} from {url}")));
        }
        let body: Value = res.json().await?;
        let docs = body["response"]["docs"]
            .as_array()
            .ok_or_else(|| ImportError::Response("missing response.docs".into()))?;
        let next = body["nextCursorMark"]
            .as_str()
            .ok_or_else(|| ImportError::Response("missing nextCursorMark".into()))?;
        if docs.is_empty() {
            break;
        }
        let mut batch = Batch::default();
        for d in docs {
            if let Value::Object(m) = d {
                batch.push_add(Value::Object(clean_doc(index.schema(), m.clone())));
            }
        }
        let generation = index.commit(batch).await?;
        imported += docs.len();
        progress(imported, generation);
        if next == cursor {
            break;
        }
        cursor = next.to_string();
    }
    Ok(imported)
}
