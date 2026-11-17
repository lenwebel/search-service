use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use object_store::memory::InMemory;
use search_service::index::Index;
use search_service::schema::{FieldType, Schema};
use search_service::server::{AppState, router};
use serde_json::{Value, json};
use std::sync::Arc;
use tower::ServiceExt;

fn app() -> axum::Router {
    let schema = Schema::new("id")
        .with_field("id", FieldType::Str, false)
        .with_field("name", FieldType::Text, false)
        .with_field("name_str", FieldType::Str, false)
        .with_field("hitpoints", FieldType::Long, false)
        .with_field("_text_", FieldType::Text, true)
        .with_copy("name", "name_str")
        .with_copy("name", "_text_");
    let index = Index::new(Arc::new(InMemory::new()), "cards", schema);
    router(Arc::new(AppState {
        core: "cards".into(),
        index,
    }))
}

async fn call(
    app: &axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(b) => {
            req = req.header("content-type", "application/json");
            Body::from(b.to_string())
        }
        None => Body::empty(),
    };
    let res = app.clone().oneshot(req.body(body).unwrap()).await.unwrap();
    let status = res.status();
    let bytes = res.into_body().collect().await.unwrap().to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

async fn seeded() -> axum::Router {
    let app = app();
    let docs = json!([
        {"id": "1", "name": "Pikachu ex", "hitpoints": 190},
        {"id": "2", "name": "Pikachu", "hitpoints": 40},
        {"id": "3", "name": "Raichu", "hitpoints": 80}
    ]);
    let (status, body) = call(&app, "POST", "/solr/cards/update?commit=true", Some(docs)).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["responseHeader"]["status"], 0);
    app
}

#[tokio::test]
async fn select_returns_solr_shaped_response() {
    let app = seeded().await;
    let (status, body) = call(
        &app,
        "GET",
        "/solr/cards/select?q=name:pikachu&fl=id,score",
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["responseHeader"]["status"], 0);
    assert_eq!(body["responseHeader"]["params"]["q"], "name:pikachu");
    assert_eq!(body["response"]["numFound"], 2);
    assert_eq!(body["response"]["start"], 0);
    assert_eq!(body["response"]["numFoundExact"], true);
    let docs = body["response"]["docs"].as_array().unwrap();
    assert_eq!(docs[0], json!({"id": "2", "score": docs[0]["score"]}));
    assert!(docs[0]["score"].as_f64().unwrap() > 0.0);
    assert!(body["response"]["maxScore"].as_f64().is_some());
}

#[tokio::test]
async fn select_supports_df_qop_fq_sort_paging_and_repeated_params() {
    let app = seeded().await;
    let uri = "/solr/cards/select?q=pikachu%20raichu&df=name&fq=hitpoints:%5B50%20TO%20*%5D&fq=-id:1&sort=hitpoints%20desc&rows=5";
    let (_, body) = call(&app, "GET", uri, None).await;
    let ids: Vec<&str> = body["response"]["docs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["3"]);
    assert!(
        body["response"].get("maxScore").is_none(),
        "maxScore only when score is requested"
    );

    let (_, body) = call(
        &app,
        "GET",
        "/solr/cards/select?q=pikachu%20ex&df=name&q.op=AND",
        None,
    )
    .await;
    assert_eq!(body["response"]["numFound"], 1);

    let (_, body) = call(
        &app,
        "GET",
        "/solr/cards/select?q=*:*&sort=name_str%20asc&start=1&rows=1",
        None,
    )
    .await;
    assert_eq!(body["response"]["numFound"], 3);
    assert_eq!(
        body["response"]["docs"][0]["id"], "1",
        "Pikachu < Pikachu ex < Raichu"
    );
}

#[tokio::test]
async fn update_handles_add_and_delete_commands() {
    let app = seeded().await;
    let cmd = json!({"delete": {"id": "2"}});
    assert_eq!(
        call(&app, "POST", "/solr/cards/update", Some(cmd)).await.0,
        StatusCode::OK
    );
    let cmd = json!({"add": {"doc": {"id": "4", "name": "Pichu"}}, "delete": ["3"]});
    assert_eq!(
        call(&app, "POST", "/solr/cards/update", Some(cmd)).await.0,
        StatusCode::OK
    );
    let (_, body) = call(
        &app,
        "GET",
        "/solr/cards/select?q=*:*&sort=id%20asc&fl=id",
        None,
    )
    .await;
    assert_eq!(body["response"]["docs"], json!([{"id": "1"}, {"id": "4"}]));
    // Default df is _text_, fed by the copy field.
    let (_, body) = call(&app, "GET", "/solr/cards/select?q=pichu", None).await;
    assert_eq!(body["response"]["numFound"], 1);
}

#[tokio::test]
async fn errors_use_solr_error_shape() {
    let app = seeded().await;
    let (status, body) = call(&app, "GET", "/solr/cards/select?q=name:(unclosed", None).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["responseHeader"]["status"], 400);
    assert_eq!(body["error"]["code"], 400);
    assert!(body["error"]["msg"].as_str().unwrap().contains(")"));

    let (status, _) = call(&app, "GET", "/solr/nope/select?q=*:*", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    let (status, body) = call(
        &app,
        "POST",
        "/solr/cards/update",
        Some(json!([{"name": "no id"}])),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"]["msg"].as_str().unwrap().contains("id"));
}

#[tokio::test]
async fn update_accepts_bulk_bodies_larger_than_two_megabytes() {
    let app = app();
    let filler = "x".repeat(1000);
    let docs: Vec<Value> = (0..5000)
        .map(|i| json!({"id": i.to_string(), "name": format!("card {i} {filler}")}))
        .collect();
    let (status, body) = call(&app, "POST", "/solr/cards/update", Some(Value::Array(docs))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let (_, body) = call(&app, "GET", "/solr/cards/select?q=*:*&rows=0", None).await;
    assert_eq!(body["response"]["numFound"], 5000);
}
