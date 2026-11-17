use object_store::ObjectStore;
use object_store::local::LocalFileSystem;
use object_store::memory::InMemory;
use search_service::index::{Batch, Index, SearchRequest, SortField};
use search_service::query::{ParseOptions, Query, parse};
use search_service::schema::{FieldType, Schema};
use serde_json::json;
use std::sync::Arc;

fn schema() -> Schema {
    Schema::new("id")
        .with_field("id", FieldType::Str, false)
        .with_field("name", FieldType::Text, false)
        .with_field("name_str", FieldType::Str, false)
        .with_field("set", FieldType::Text, false)
        .with_field("hitpoints", FieldType::Long, false)
        .with_field("price", FieldType::Double, false)
        .with_field("released", FieldType::Date, false)
        .with_field("search_all", FieldType::Text, true)
        .with_copy("name", "name_str")
        .with_copy("name", "search_all")
        .with_copy("set", "search_all")
}

fn q(s: &str) -> Query {
    parse(
        s,
        &schema(),
        &ParseOptions {
            default_field: "search_all".into(),
            default_and: false,
        },
    )
    .unwrap()
}

fn req(query: &str) -> SearchRequest {
    SearchRequest {
        query: q(query),
        filters: vec![],
        sort: vec![],
        start: 0,
        rows: 100,
    }
}

fn mem_index() -> Index {
    Index::new(Arc::new(InMemory::new()), "cards", schema())
}

async fn ids(index: &Index, r: SearchRequest) -> Vec<String> {
    let res = index.searcher().await.unwrap().search(&r).unwrap();
    res.docs
        .iter()
        .map(|h| h.doc["id"].as_str().unwrap().to_string())
        .collect()
}

async fn seed(index: &Index) {
    index
        .commit(
            Batch::default()
                .add(json!({"id": "1", "name": "Pikachu ex", "set": "Scarlet Violet", "hitpoints": 190, "released": "2023-03-31T00:00:00Z"}))
                .add(json!({"id": "2", "name": "Pikachu", "set": "Base Set", "hitpoints": 40, "released": "1999-01-09T00:00:00Z"}))
                .add(json!({"id": "3", "name": "Raichu", "set": "Base Set", "hitpoints": 80, "released": "1999-01-09T00:00:00Z"})),
        )
        .await
        .unwrap();
}

#[tokio::test]
async fn committed_documents_are_searchable_by_term_phrase_exact_and_range() {
    let index = mem_index();
    seed(&index).await;
    let mut hits = ids(&index, req("pikachu")).await;
    hits.sort();
    assert_eq!(hits, vec!["1", "2"]);
    assert_eq!(ids(&index, req("set:\"base set\"")).await.len(), 2);
    assert_eq!(ids(&index, req("name_str:\"Pikachu ex\"")).await, vec!["1"]);
    assert_eq!(ids(&index, req("name_str:Pika*")).await.len(), 2);
    assert_eq!(ids(&index, req("hitpoints:[50 TO 200]")).await.len(), 2);
    assert_eq!(
        ids(&index, req("released:[2000-01-01T00:00:00Z TO *]")).await,
        vec!["1"]
    );
    assert_eq!(ids(&index, req("pikachu -set:base")).await, vec!["1"]);
}

#[tokio::test]
async fn bm25_ranks_shorter_field_match_higher_and_stored_doc_round_trips() {
    let index = mem_index();
    seed(&index).await;
    let res = index
        .searcher()
        .await
        .unwrap()
        .search(&req("name:pikachu"))
        .unwrap();
    assert_eq!(res.num_found, 2);
    assert_eq!(
        res.docs[0].doc["id"], "2",
        "'Pikachu' is shorter than 'Pikachu ex'"
    );
    assert!(res.docs[0].score > res.docs[1].score);
    assert_eq!(res.docs[0].doc["hitpoints"], 40);
    assert!(
        res.docs[0].doc.get("name_str").is_none(),
        "copy targets are not stored"
    );
}

#[tokio::test]
async fn filters_sort_and_paging() {
    let index = mem_index();
    seed(&index).await;
    let r = SearchRequest {
        query: Query::MatchAll,
        filters: vec![q("set:base")],
        sort: vec![SortField {
            field: "hitpoints".into(),
            descending: true,
        }],
        start: 0,
        rows: 1,
    };
    let res = index.searcher().await.unwrap().search(&r).unwrap();
    assert_eq!(res.num_found, 2);
    assert_eq!(res.docs.len(), 1);
    assert_eq!(res.docs[0].doc["id"], "3");
    let r = SearchRequest {
        start: 1,
        sort: vec![SortField {
            field: "name_str".into(),
            descending: false,
        }],
        ..r
    };
    let res = index.searcher().await.unwrap().search(&r).unwrap();
    assert_eq!(
        res.docs[0].doc["id"], "3",
        "Pikachu < Raichu, page 2 is Raichu"
    );
}

#[tokio::test]
async fn delete_and_readd_follow_commit_order() {
    let index = mem_index();
    seed(&index).await;
    index.commit(Batch::default().delete("2")).await.unwrap();
    assert_eq!(ids(&index, req("name:pikachu")).await, vec!["1"]);

    // Re-adding an id replaces the older version (Solr overwrite semantics).
    index
        .commit(Batch::default().add(json!({"id": "1", "name": "Pikachu V"})))
        .await
        .unwrap();
    let res = index
        .searcher()
        .await
        .unwrap()
        .search(&req("name:pikachu"))
        .unwrap();
    assert_eq!(res.num_found, 1);
    assert_eq!(res.docs[0].doc["name"], "Pikachu V");

    // Within one batch, operations apply in order.
    index
        .commit(
            Batch::default()
                .delete("3")
                .add(json!({"id": "3", "name": "Raichu again"})),
        )
        .await
        .unwrap();
    assert_eq!(ids(&index, req("raichu")).await, vec!["3"]);
    index
        .commit(
            Batch::default()
                .add(json!({"id": "4", "name": "Mew"}))
                .delete("4"),
        )
        .await
        .unwrap();
    assert!(ids(&index, req("mew")).await.is_empty());
}

#[tokio::test]
async fn rejects_documents_without_unique_key_or_with_bad_numbers() {
    let index = mem_index();
    assert!(
        index
            .commit(Batch::default().add(json!({"name": "no id"})))
            .await
            .is_err()
    );
    assert!(
        index
            .commit(Batch::default().add(json!({"id": "x", "hitpoints": "lots"})))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn concurrent_writers_on_shared_storage_lose_no_commits() {
    let dir = tempfile::tempdir().unwrap();
    let store_a: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    let store_b: Arc<dyn ObjectStore> =
        Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
    // Two independent "instances" pointing at the same storage, no shared memory.
    let a = Arc::new(Index::new(store_a, "cards", schema()));
    let b = Arc::new(Index::new(store_b, "cards", schema()));

    let mut tasks = vec![];
    for w in 0..8 {
        let index = if w % 2 == 0 { a.clone() } else { b.clone() };
        tasks.push(tokio::spawn(async move {
            for i in 0..5 {
                let id = format!("w{w}-{i}");
                index
                    .commit(Batch::default().add(json!({"id": id, "name": "card"})))
                    .await
                    .unwrap();
            }
        }));
    }
    for t in tasks {
        t.await.unwrap();
    }
    let searcher = a.searcher().await.unwrap();
    assert_eq!(
        searcher.generation(),
        40,
        "every commit got its own generation"
    );
    assert_eq!(searcher.search(&req("card")).unwrap().num_found, 40);
    assert_eq!(
        b.searcher()
            .await
            .unwrap()
            .search(&req("card"))
            .unwrap()
            .num_found,
        40
    );
}

#[tokio::test]
async fn corrupted_segment_is_rejected_not_served() {
    use futures::TryStreamExt;
    use object_store::{ObjectStoreExt, PutPayload, path::Path};
    let store = Arc::new(InMemory::new());
    let index = Index::new(store.clone(), "cards", schema());
    seed(&index).await;
    let segs: Vec<_> = store
        .list(Some(&Path::from("cards/segments")))
        .try_collect()
        .await
        .unwrap();
    let path = &segs[0].location;
    let mut bytes = store
        .get(path)
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap()
        .to_vec();
    let mid = bytes.len() / 2;
    bytes[mid] ^= 0x01;
    store.put(path, PutPayload::from(bytes)).await.unwrap();
    // A fresh instance has nothing cached, so it must download and verify.
    let fresh = Index::new(store.clone(), "cards", schema());
    let err = fresh
        .searcher()
        .await
        .err()
        .expect("corruption must be detected");
    assert!(err.to_string().contains("checksum"), "{err}");
}

#[tokio::test]
async fn sorting_and_phrases_work_across_segments_and_values() {
    let index = mem_index();
    index
        .commit(
            Batch::default().add(
                json!({"id": "b", "name": "Bulbasaur", "search_all": ["grass starter", "kanto"]}),
            ),
        )
        .await
        .unwrap();
    index
        .commit(Batch::default().add(json!({"id": "a", "name": "Abra"})))
        .await
        .unwrap();
    index
        .commit(Batch::default().add(json!({"id": "c", "name": "Charmander"})))
        .await
        .unwrap();
    let r = SearchRequest {
        query: Query::MatchAll,
        filters: vec![],
        sort: vec![SortField {
            field: "name_str".into(),
            descending: false,
        }],
        start: 0,
        rows: 10,
    };
    let names: Vec<String> = index
        .searcher()
        .await
        .unwrap()
        .search(&r)
        .unwrap()
        .docs
        .iter()
        .map(|h| h.doc["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(names, vec!["a", "b", "c"]);
    assert_eq!(
        ids(&index, req("search_all:\"grass starter\"")).await,
        vec!["b"]
    );
    assert!(
        ids(&index, req("search_all:\"starter kanto\""))
            .await
            .is_empty(),
        "no phrase match across values"
    );
}

#[tokio::test]
async fn repeated_filters_stay_correct_across_new_commits_and_deletes() {
    let index = mem_index();
    seed(&index).await;
    let filtered = |f: &str| SearchRequest {
        query: Query::MatchAll,
        filters: vec![q(f)],
        sort: vec![],
        start: 0,
        rows: 100,
    };
    let count = |r: SearchRequest| {
        let index = &index;
        async move {
            index
                .searcher()
                .await
                .unwrap()
                .search(&r)
                .unwrap()
                .num_found
        }
    };
    assert_eq!(count(filtered("set:base")).await, 2);
    assert_eq!(
        count(filtered("set:base")).await,
        2,
        "served from cache, same answer"
    );
    index
        .commit(Batch::default().add(json!({"id": "9", "name": "Clefairy", "set": "Base Set"})))
        .await
        .unwrap();
    assert_eq!(
        count(filtered("set:base")).await,
        3,
        "filter applies to the new segment"
    );
    index.commit(Batch::default().delete("2")).await.unwrap();
    assert_eq!(
        count(filtered("set:base")).await,
        2,
        "deleted doc drops out despite cached filter"
    );
    let r = SearchRequest {
        query: q("raichu OR clefairy"),
        filters: vec![q("set:base")],
        sort: vec![],
        start: 0,
        rows: 100,
    };
    assert_eq!(
        index
            .searcher()
            .await
            .unwrap()
            .search(&r)
            .unwrap()
            .num_found,
        2
    );
}

#[tokio::test]
async fn field_list_limits_returned_fields() {
    let index = mem_index();
    seed(&index).await;
    let searcher = index.searcher().await.unwrap();
    let res = searcher
        .search_fields(&req("name:raichu"), Some(&["id"]))
        .unwrap();
    assert_eq!(
        res.docs[0].doc,
        serde_json::from_value::<serde_json::Map<String, serde_json::Value>>(json!({"id": "3"}))
            .unwrap()
    );
    let res = searcher
        .search_fields(&req("name:raichu"), Some(&["name", "hitpoints"]))
        .unwrap();
    assert_eq!(res.docs[0].doc.len(), 2);
    assert_eq!(res.docs[0].doc["hitpoints"], 80);
}

#[tokio::test]
async fn cached_searcher_avoids_storage_until_refreshed() {
    let store = Arc::new(InMemory::new());
    let writer = Index::new(store.clone(), "cards", schema());
    let reader = Index::new(store.clone(), "cards", schema());
    seed(&writer).await;
    assert_eq!(
        reader.cached_searcher().await.unwrap().num_docs(),
        3,
        "first use opens a searcher"
    );
    writer
        .commit(Batch::default().add(json!({"id": "9", "name": "Clefairy"})))
        .await
        .unwrap();
    assert_eq!(
        reader.cached_searcher().await.unwrap().num_docs(),
        3,
        "no storage check per request"
    );
    assert_eq!(
        reader.searcher().await.unwrap().num_docs(),
        4,
        "refresh picks up other writers"
    );
    assert_eq!(reader.cached_searcher().await.unwrap().num_docs(), 4);
}

#[tokio::test]
async fn searches_under_load_return_the_same_results_as_a_lone_search() {
    let index = mem_index();
    for batch in 0..6 {
        let docs = (0..40).map(|i| {
            json!({
                "id": format!("{batch}-{i}"),
                "name": format!("card {} {}", ["fire", "water", "grass"][i % 3], i % 7),
                "set": format!("set {}", batch % 2),
                "hitpoints": (i * 10) % 170,
            })
        });
        let mut b = Batch::default();
        for d in docs {
            b.push_add(d);
        }
        index.commit(b).await.unwrap();
    }
    let searcher = index.searcher().await.unwrap();
    let requests: Vec<SearchRequest> = vec![
        req("fire OR water"),
        req("name:\"card fire\""),
        SearchRequest {
            sort: vec![SortField {
                field: "hitpoints".into(),
                descending: true,
            }],
            ..req("*:*")
        },
        SearchRequest {
            filters: vec![q("set:1")],
            sort: vec![SortField {
                field: "name_str".into(),
                descending: false,
            }],
            ..req("grass")
        },
    ];
    let summarise = |r: &SearchRequest| {
        let res = searcher.search(r).unwrap();
        let hits: Vec<(String, u32)> = res
            .docs
            .iter()
            .map(|h| (h.doc["id"].as_str().unwrap().to_string(), h.score.to_bits()))
            .collect();
        (res.num_found, hits)
    };
    let alone: Vec<_> = requests.iter().map(summarise).collect();

    // Many searches in flight at once, as on a busy server.
    let _load: Vec<_> = (0..8)
        .map(|_| search_service::search::Pending::enter())
        .collect();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| s.spawn(|| requests.iter().map(summarise).collect::<Vec<_>>()))
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), alone);
        }
    });
}

#[tokio::test]
async fn sorted_results_match_a_reference_sort_across_segments() {
    // Values chosen to hit the edge cases: missing values, duplicates (ties
    // fall back to index order), negative numbers, and -0.0 against 0.0.
    let names = [
        Some("b"),
        None,
        Some("a"),
        Some("c"),
        Some("a"),
        None,
        Some("B"),
    ];
    let hps = [
        Some(30),
        Some(-5),
        None,
        Some(30),
        Some(0),
        Some(-5),
        None,
        Some(120),
    ];
    let prices = [
        Some(1.5),
        Some(-0.0),
        Some(0.0),
        None,
        Some(-2.25),
        Some(1.5),
    ];
    let mut docs = vec![];
    for i in 0..60 {
        let mut d = serde_json::Map::new();
        d.insert("id".into(), json!(format!("d{i:02}")));
        if let Some(n) = names[i % names.len()] {
            d.insert("name_str".into(), json!(n));
        }
        if let Some(h) = hps[i % hps.len()] {
            d.insert("hitpoints".into(), json!(h));
        }
        if let Some(p) = prices[i % prices.len()] {
            d.insert("price".into(), json!(p));
        }
        docs.push(serde_json::Value::Object(d));
    }

    // Missing values sort last in both directions; ties keep index order.
    let reference = |specs: &[(&str, bool)]| -> Vec<String> {
        let mut order: Vec<usize> = (0..docs.len()).collect();
        order.sort_by(|&a, &b| {
            for &(field, desc) in specs {
                let (va, vb) = (&docs[a][field], &docs[b][field]);
                let o = match (va.is_null(), vb.is_null()) {
                    (true, true) => std::cmp::Ordering::Equal,
                    (true, false) => return std::cmp::Ordering::Greater,
                    (false, true) => return std::cmp::Ordering::Less,
                    _ => match (va.as_str(), vb.as_str()) {
                        (Some(x), Some(y)) => x.cmp(y),
                        _ => va
                            .as_f64()
                            .unwrap()
                            .partial_cmp(&vb.as_f64().unwrap())
                            .unwrap(),
                    },
                };
                let o = if desc { o.reverse() } else { o };
                if o.is_ne() {
                    return o;
                }
            }
            a.cmp(&b)
        });
        order
            .iter()
            .map(|&i| docs[i]["id"].as_str().unwrap().to_string())
            .collect()
    };

    let cases: Vec<Vec<(&str, bool)>> = vec![
        vec![("name_str", false)],
        vec![("name_str", true)],
        vec![("hitpoints", true), ("name_str", false)],
        vec![("price", false)],
        vec![("price", true), ("hitpoints", false), ("name_str", true)],
    ];
    // Many small segments, and one segment where a page cut can fall between
    // equal values.
    for segment_size in [13, 60] {
        let index = mem_index();
        for chunk in docs.chunks(segment_size) {
            let mut b = Batch::default();
            for d in chunk {
                b.push_add(d.clone());
            }
            index.commit(b).await.unwrap();
        }
        for specs in &cases {
            let expected = reference(specs);
            for (start, rows) in [(0, 60), (0, 7), (11, 9), (0, 3), (10, 3), (10, 5)] {
                let r = SearchRequest {
                    sort: specs
                        .iter()
                        .map(|&(f, d)| SortField {
                            field: f.into(),
                            descending: d,
                        })
                        .collect(),
                    start,
                    rows,
                    ..req("*:*")
                };
                assert_eq!(
                    ids(&index, r).await,
                    expected[start..start + rows].to_vec(),
                    "sort {specs:?}, start {start}, rows {rows}, segments of {segment_size}"
                );
            }
        }
    }
}

#[tokio::test]
async fn wildcards_match_exactly_what_a_reference_matcher_matches() {
    // Brute-force glob over chars: `*` is any run, `?` exactly one char.
    fn reference(p: &[char], t: &[char]) -> bool {
        match (p.first(), t.first()) {
            (None, _) => t.is_empty(),
            (Some('*'), _) => reference(&p[1..], t) || (!t.is_empty() && reference(p, &t[1..])),
            (Some('?'), Some(_)) => reference(&p[1..], &t[1..]),
            (Some(c), Some(d)) if c == d => reference(&p[1..], &t[1..]),
            _ => false,
        }
    }
    let values = [
        "pikachu",
        "charizard",
        "aaa",
        "aab",
        "baa",
        "abab",
        "ébène",
        "naïve",
        "zz",
        "a",
        "chachacha",
        "ab",
    ];
    let index = mem_index();
    let mut b = Batch::default();
    for (i, v) in values.iter().enumerate() {
        b.push_add(json!({"id": format!("w{i}"), "name_str": v}));
    }
    index.commit(b).await.unwrap();
    let patterns = [
        "*a*",
        "*ar*d",
        "c*z*d",
        "*aa*a",
        "a?a",
        "?a?",
        "*é*",
        "?bène",
        "a**b",
        "*u",
        "*ch*",
        "*cha*cha",
        "*b*b*",
        "*ab",
        "?",
        "??",
        "*?",
        "na?ve",
        "*cha*a*cha*",
    ];
    for p in patterns {
        let pc: Vec<char> = p.chars().collect();
        let mut expected: Vec<String> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| reference(&pc, &v.chars().collect::<Vec<_>>()))
            .map(|(i, _)| format!("w{i}"))
            .collect();
        let mut got = ids(&index, req(&format!("name_str:{p}"))).await;
        expected.sort();
        got.sort();
        assert_eq!(got, expected, "pattern {p}");
    }
}
