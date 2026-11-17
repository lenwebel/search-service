use object_store::ObjectStore;
use search_service::import::{ImportOptions, import_from_solr};
use search_service::index::Index;
use search_service::schema::Schema;
use search_service::server::{AppState, router};
use std::sync::Arc;

const USAGE: &str = "\
usage:
  search-service serve
  search-service import --from <solr core url> [--page-size N] [--limit N]

environment:
  STORE_URL   where the index lives: file:///abs/path or az://container/path
              (Azure credentials come from AZURE_STORAGE_* variables)
  CORE        core name served at /solr/<CORE>/ (default: collectorspoke-cards)
  SCHEMA_DIR  directory holding a Solr schema export (default: schemas/<CORE>)
  BIND        listen address for serve (default: 127.0.0.1:8983)
  REFRESH_MS  how often serve checks storage for other instances' commits (default: 1000)
  SEGMENT_CACHE_DIR  local directory for verified segment files (default: a temp dir)
  SOLR_AUTH_CREDENTIALS  base64 user:password for import, optional";

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.to_string())
}

fn open_store(store_url: &str) -> Result<(Arc<dyn ObjectStore>, String), BoxError> {
    let url = url::Url::parse(store_url)?;
    if url.scheme() == "file" {
        std::fs::create_dir_all(url.path())?;
    }
    let azure_opts = std::env::vars()
        .filter(|(k, _)| k.starts_with("AZURE_"))
        .map(|(k, v)| (k.to_ascii_lowercase(), v));
    let (store, path) = object_store::parse_url_opts(&url, azure_opts)?;
    Ok((Arc::from(store), path.to_string()))
}

fn open_index() -> Result<(Index, String), BoxError> {
    let core = env_or("CORE", "collectorspoke-cards");
    let schema = Schema::from_solr_dir(env_or("SCHEMA_DIR", &format!("schemas/{core}")))?;
    let store_url = std::env::var("STORE_URL").map_err(|_| "STORE_URL is required")?;
    let (store, base) = open_store(&store_url)?;
    let prefix = if base.is_empty() {
        core.clone()
    } else {
        format!("{base}/{core}")
    };
    let index = match std::env::var("SEGMENT_CACHE_DIR") {
        Ok(dir) => Index::with_cache_dir(store, &prefix, schema, dir.into()),
        Err(_) => Index::new(store, &prefix, schema),
    };
    Ok((index, core))
}

fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("serve") => {
            let (index, core) = open_index()?;
            let bind = env_or("BIND", "127.0.0.1:8983");
            let listener = tokio::net::TcpListener::bind(&bind).await?;
            println!("serving /solr/{core}/ on http://{bind}");
            let state = Arc::new(AppState { core, index });
            // Pick up commits from other instances without a storage round trip per query.
            let refresh = std::time::Duration::from_millis(env_or("REFRESH_MS", "1000").parse()?);
            let bg = state.clone();
            tokio::spawn(async move {
                loop {
                    if let Err(e) = bg.index.searcher().await {
                        eprintln!("refresh failed: {e}");
                    }
                    tokio::time::sleep(refresh).await;
                }
            });
            axum::serve(listener, router(state)).await?;
        }
        Some("import") => {
            let (index, _) = open_index()?;
            let opts = ImportOptions {
                core_url: flag(&args, "--from").ok_or(USAGE)?,
                authorization: std::env::var("SOLR_AUTH_CREDENTIALS")
                    .ok()
                    .map(|c| format!("Basic {c}")),
                page_size: flag(&args, "--page-size").map_or(Ok(5000), |v| v.parse())?,
                limit: flag(&args, "--limit").map(|v| v.parse()).transpose()?,
            };
            let started = std::time::Instant::now();
            let n = import_from_solr(&index, &opts, |done, generation| {
                println!(
                    "imported {done} documents (generation {generation}, {:.1?})",
                    started.elapsed()
                );
            })
            .await?;
            println!("done: {n} documents in {:.1?}", started.elapsed());
        }
        _ => {
            eprintln!("{USAGE}");
            std::process::exit(2);
        }
    }
    Ok(())
}
