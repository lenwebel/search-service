//! Benchmarks against a real index.
//!
//!   STORE_URL=file:///... cargo run --release --example bench -- queries <out.jsonl>
//!   STORE_URL=file:///... cargo run --release --example bench -- engine <queries.jsonl>
//!   cargo run --release --example bench -- http <base select url> <queries.jsonl> [concurrency] [per-class]
//!   STORE_URL=file:///... cargo run --release --example bench -- index <docs> <batch>
//!
//! `queries` derives a realistic query set from the indexed cards so every
//! engine (this one, in-process or over HTTP, and Solr) runs identical queries.

use object_store::local::LocalFileSystem;
use search_service::index::{Batch, Index, SearchRequest, SortField};
use search_service::query::{ParseOptions, Query, parse};
use search_service::schema::Schema;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

fn rss_mb() -> f64 {
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .unwrap();
    String::from_utf8_lossy(&out.stdout)
        .trim()
        .parse::<f64>()
        .unwrap_or(0.0)
        / 1024.0
}

fn open_index() -> Result<Index, BoxError> {
    let core = std::env::var("CORE").unwrap_or("collectorspoke-cards".into());
    let schema = Schema::from_solr_dir(format!("schemas/{core}"))?;
    let url = url::Url::parse(&std::env::var("STORE_URL")?)?;
    let (store, base) = object_store::parse_url(&url)?;
    Ok(Index::new(
        Arc::from(store),
        &format!("{base}/{core}"),
        schema,
    ))
}

fn pct(sorted: &[Duration], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let i = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[i].as_secs_f64() * 1000.0
}

fn summary(label: &str, mut lat: Vec<Duration>) {
    lat.sort();
    println!(
        "{label:<18} n={:<5} p50={:>7.2}ms p95={:>7.2}ms p99={:>7.2}ms max={:>8.2}ms",
        lat.len(),
        pct(&lat, 0.5),
        pct(&lat, 0.95),
        pct(&lat, 0.99),
        lat.last().map_or(0.0, |d| d.as_secs_f64() * 1000.0)
    );
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn words(s: &str) -> Vec<String> {
    s.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_string())
        .filter(|w| w.len() >= 3)
        .collect()
}

/// Query classes modelled on what the app sends (see tradingcardgames/src/lib/solr.ts).
async fn gen_queries(out: &str) -> Result<(), BoxError> {
    let index = open_index()?;
    let searcher = index.searcher().await?;
    let total = searcher.num_docs();
    let sample = searcher.search(&SearchRequest {
        query: Query::MatchAll,
        filters: vec![],
        sort: vec![],
        start: 0,
        rows: total,
    })?;
    let docs: Vec<&serde_json::Map<String, Value>> = sample
        .docs
        .iter()
        .map(|h| &h.doc)
        .step_by((total / 400).max(1))
        .collect();
    let s = |d: &serde_json::Map<String, Value>, k: &str| {
        d.get(k).and_then(Value::as_str).unwrap_or("").to_string()
    };
    let mut lines = vec![];
    for (i, d) in docs.iter().enumerate() {
        let (name, brand, set, rarity) = (s(d, "name"), s(d, "brand"), s(d, "set"), s(d, "rarity"));
        let ws = words(&name);
        if ws.is_empty() || brand.is_empty() {
            continue;
        }
        let w0 = &ws[0];
        let mut add = |class: &str, p: Value| lines.push(json!({"class": class, "params": p}));
        // 1. The app's card lookup: constant-score boosted OR across exact fields.
        add(
            "app_lookup",
            json!({"q": format!(
            "(brand_str:\"{}\")^=100 OR (name_str:\"{}*\")^=10 OR (name_str:\"{}\")^=50 OR (set_str:\"{}\")^=20 OR (rarity_str:\"{}\")^=10",
            escape(&brand), escape(w0), escape(&name), escape(&set), escape(&rarity)), "rows": "50"}),
        );
        // 2. Marketplace free text against the catch-all field.
        add(
            "free_text",
            json!({"q": ws.iter().take(2).cloned().collect::<Vec<_>>().join(" "), "df": "search_all", "rows": "50"}),
        );
        // 3. Phrase.
        if ws.len() >= 2 {
            add(
                "phrase",
                json!({"q": format!("name:\"{} {}\"", ws[0], ws[1]), "rows": "10"}),
            );
        }
        // 4. Prefix on the exact-name field (what the app meant to send).
        add(
            "prefix",
            json!({"q": format!("name_str:{}*", w0.chars().take(4).collect::<String>()), "rows": "10"}),
        );
        // 5. Filtered + sorted browse.
        add(
            "filter_sort",
            json!({"q": format!("search_all:{}", ws.last().unwrap()), "fq": format!("brand_str:\"{}\"", escape(&brand)),
            "sort": "releaseDate_str desc, name_str asc", "rows": "50"}),
        );
        // 6. Infix wildcard: a full term-dictionary scan.
        if i % 4 == 0 && w0.len() >= 5 {
            add(
                "infix_wildcard",
                json!({"q": format!("name_str:*{}*", &w0[1..4]), "rows": "10"}),
            );
        }
        // 7. Browse everything, sorted: touches every document.
        if i % 4 == 0 {
            add(
                "match_all_sort",
                json!({"q": "*:*", "fq": format!("brand_str:\"{}\"", escape(&brand)), "sort": "name_str asc", "rows": "50", "start": (i % 20 * 50).to_string()}),
            );
        }
    }
    let body: String = lines.iter().map(|l| l.to_string() + "\n").collect();
    std::fs::write(out, body)?;
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    lines.iter().for_each(|l| {
        *counts
            .entry(l["class"].as_str().unwrap().to_string())
            .or_default() += 1
    });
    println!("wrote {} queries to {out}: {counts:?}", lines.len());
    Ok(())
}

type QuerySet = Vec<(String, Vec<(String, String)>)>;

fn load_queries(path: &str) -> Result<QuerySet, BoxError> {
    let text = std::fs::read_to_string(path)?;
    Ok(text
        .lines()
        .map(|l| {
            let v: Value = serde_json::from_str(l).unwrap();
            let params = v["params"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.as_str().unwrap().to_string()))
                .collect();
            (v["class"].as_str().unwrap().to_string(), params)
        })
        .collect())
}

fn to_request(schema: &Schema, params: &[(String, String)]) -> SearchRequest {
    let get = |k: &str| {
        params
            .iter()
            .find(|(pk, _)| pk == k)
            .map(|(_, v)| v.as_str())
    };
    let opts = ParseOptions {
        default_field: get("df").unwrap_or("_text_").into(),
        default_and: get("q.op") == Some("AND"),
    };
    let sort = get("sort")
        .unwrap_or("")
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            let mut p = s.split_whitespace();
            SortField {
                field: p.next().unwrap().into(),
                descending: p.next() == Some("desc"),
            }
        })
        .collect();
    SearchRequest {
        query: parse(get("q").unwrap(), schema, &opts).unwrap(),
        filters: params
            .iter()
            .filter(|(k, _)| k == "fq")
            .map(|(_, f)| parse(f, schema, &opts).unwrap())
            .collect(),
        sort,
        start: get("start").map_or(0, |s| s.parse().unwrap()),
        rows: get("rows").map_or(10, |s| s.parse().unwrap()),
    }
}

async fn engine(path: &str) -> Result<(), BoxError> {
    let rss0 = rss_mb();
    let index = open_index()?;
    let t = Instant::now();
    let searcher = index.searcher().await?;
    println!(
        "cold open: {:.2?} for {} docs, generation {}, RSS {:.0} MB -> {:.0} MB",
        t.elapsed(),
        searcher.num_docs(),
        searcher.generation(),
        rss0,
        rss_mb()
    );
    let t = Instant::now();
    let warm = index.searcher().await?;
    println!("warm open (manifest probe, cached): {:.2?}", t.elapsed());
    drop(warm);

    let queries = load_queries(path)?;
    let reqs: Vec<(String, SearchRequest)> = queries
        .iter()
        .map(|(c, p)| (c.clone(), to_request(index.schema(), p)))
        .collect();
    // Warm-up pass, then measured single-threaded pass.
    for (_, r) in &reqs {
        searcher.search(r)?;
    }
    let mut by_class: BTreeMap<String, Vec<Duration>> = BTreeMap::new();
    let mut found: BTreeMap<String, usize> = BTreeMap::new();
    for (c, r) in &reqs {
        let t = Instant::now();
        let res = searcher.search(r)?;
        by_class.entry(c.clone()).or_default().push(t.elapsed());
        *found.entry(c.clone()).or_default() += res.num_found;
    }
    println!("\nsingle-threaded latency (in-process, no HTTP):");
    for (c, lat) in by_class {
        let n = lat.len();
        summary(&c, lat);
        println!("{:<18} avg hits={}", "", found[&c] / n.max(1));
    }

    for threads in [1, 4, 8, 14] {
        let t = Instant::now();
        let window = Duration::from_secs(5);
        let done = std::sync::atomic::AtomicUsize::new(0);
        std::thread::scope(|s| {
            for w in 0..threads {
                let (searcher, reqs, done) = (&searcher, &reqs, &done);
                s.spawn(move || {
                    let mut i = w * 7;
                    while t.elapsed() < window {
                        searcher.search(&reqs[i % reqs.len()].1).unwrap();
                        done.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        i += 1;
                    }
                });
            }
        });
        let n = done.into_inner();
        println!(
            "throughput, {threads:>2} threads, mixed queries: {:>7.0} q/s",
            n as f64 / t.elapsed().as_secs_f64()
        );
    }
    println!("RSS after queries: {:.0} MB", rss_mb());
    Ok(())
}

async fn http(
    base: &str,
    path: &str,
    concurrency: usize,
    per_class: usize,
) -> Result<(), BoxError> {
    let auth = std::env::var("SOLR_AUTH_CREDENTIALS")
        .ok()
        .map(|c| format!("Basic {c}"));
    let client = reqwest::Client::new();
    let mut queries = load_queries(path)?;
    let mut seen: BTreeMap<String, usize> = BTreeMap::new();
    queries.retain(|(c, _)| {
        let n = seen.entry(c.clone()).or_default();
        *n += 1;
        *n <= per_class
    });
    let queries = Arc::new(queries);
    let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let started = Instant::now();
    let mut tasks = vec![];
    for _ in 0..concurrency {
        let (client, queries, next, auth, base) = (
            client.clone(),
            queries.clone(),
            next.clone(),
            auth.clone(),
            base.to_string(),
        );
        tasks.push(tokio::spawn(async move {
            let mut out: Vec<(String, Duration, u64, u64)> = vec![];
            loop {
                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some((class, params)) = queries.get(i) else {
                    break;
                };
                let mut p = params.clone();
                p.push(("fl".into(), "id".into()));
                let mut req = client.get(&base).query(&p);
                if let Some(a) = &auth {
                    req = req.header("Authorization", a);
                }
                let t = Instant::now();
                let body: Value = req.send().await.unwrap().json().await.unwrap();
                let elapsed = t.elapsed();
                let qtime = body["responseHeader"]["QTime"].as_u64().unwrap_or(u64::MAX);
                let found = body["response"]["numFound"].as_u64().unwrap_or(u64::MAX);
                out.push((class.clone(), elapsed, qtime, found));
            }
            out
        }));
    }
    let mut all = vec![];
    for t in tasks {
        all.extend(t.await?);
    }
    let wall = started.elapsed();
    println!(
        "{} requests, concurrency {concurrency}, {:.1?} wall, {:.0} req/s",
        all.len(),
        wall,
        all.len() as f64 / wall.as_secs_f64()
    );
    let mut classes: BTreeMap<String, (Vec<Duration>, Vec<Duration>, u64)> = BTreeMap::new();
    for (c, e, q, f) in all {
        let entry = classes.entry(c).or_default();
        entry.0.push(e);
        entry.1.push(Duration::from_millis(q));
        entry.2 += f;
    }
    for (c, (client_lat, server_lat, found)) in classes {
        let n = client_lat.len();
        summary(&format!("{c} client"), client_lat);
        summary(&format!("{c} QTime"), server_lat);
        println!("{:<18} total numFound={found} over {n}", "");
    }
    Ok(())
}

async fn index_bench(n_docs: usize, batch: usize) -> Result<(), BoxError> {
    let src = open_index()?;
    let searcher = src.searcher().await?;
    let docs: Vec<Value> = searcher
        .search(&SearchRequest {
            query: Query::MatchAll,
            filters: vec![],
            sort: vec![],
            start: 0,
            rows: n_docs,
        })?
        .docs
        .into_iter()
        .map(|h| Value::Object(h.doc))
        .collect();
    let docs = Arc::new(docs);
    println!(
        "indexing {} docs in batches of {batch} (local filesystem store)",
        docs.len()
    );
    for writers in [1, 4, 8] {
        let dir = tempfile::tempdir()?;
        let store = Arc::new(LocalFileSystem::new_with_prefix(dir.path())?);
        // Each writer is an independent Index instance, as separate processes would be.
        let indexes: Vec<Arc<Index>> = (0..writers)
            .map(|_| Arc::new(Index::new(store.clone(), "bench", src.schema().clone())))
            .collect();
        let next = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let t = Instant::now();
        let mut tasks = vec![];
        for index in &indexes {
            let index = index.clone();
            let (docs, next) = (docs.clone(), next.clone());
            tasks.push(tokio::spawn(async move {
                let mut lat = vec![];
                loop {
                    let start = next.fetch_add(batch, std::sync::atomic::Ordering::Relaxed);
                    if start >= docs.len() {
                        break;
                    }
                    let mut b = Batch::default();
                    docs[start..(start + batch).min(docs.len())]
                        .iter()
                        .for_each(|d| b.push_add(d.clone()));
                    let t = Instant::now();
                    index.commit(b).await.unwrap();
                    lat.push(t.elapsed());
                }
                lat
            }));
        }
        let mut lat = vec![];
        for task in tasks {
            lat.extend(task.await?);
        }
        let wall = t.elapsed();
        let check = indexes[0].searcher().await?;
        print!(
            "{writers} writer(s): {:>7.0} docs/s, {} commits, generation {}, live docs {} | ",
            docs.len() as f64 / wall.as_secs_f64(),
            lat.len(),
            check.generation(),
            check.num_docs()
        );
        summary("commit latency", lat);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    let a: Vec<String> = std::env::args().skip(1).collect();
    match a.first().map(String::as_str) {
        Some("queries") => gen_queries(&a[1]).await,
        Some("engine") => engine(&a[1]).await,
        Some("http") => {
            let c = a.get(3).map_or(1, |v| v.parse().unwrap());
            let per = a.get(4).map_or(usize::MAX, |v| v.parse().unwrap());
            http(&a[1], &a[2], c, per).await
        }
        Some("index") => index_bench(a[1].parse()?, a[2].parse()?).await,
        _ => Err("usage: bench queries|engine|http|index ...".into()),
    }
}
