//! Lock-free, multi-writer index storage on any object store.
//!
//! Layout under `<prefix>/`:
//! - `segments/<ulid>.seg`  immutable segment, written once by one writer
//! - `deletes/<ulid>.json`  ids that a commit deleted or replaced
//! - `manifest/<generation>.json`  the full list of live segments and delete files
//!
//! A commit uploads its segment and delete file under unique names (no
//! coordination needed), then creates manifest `N+1` with put-if-absent. If
//! another writer created `N+1` first, it re-reads `N+1` and tries `N+2`.
//! Nothing is ever locked, so a writer that dies leaves only unreferenced
//! files behind, never a stuck lock.
//!
//! Ordering: a delete (or re-add) of an id committed at generation G hides
//! copies of that id in segments from generations before G.
//!
//! Reading: each instance downloads a segment once, verifies its checksum,
//! writes it to a local cache directory and memory-maps it. A new searcher
//! reuses the previous one's segments and liveness, applying only the delete
//! files committed since.

use crate::schema::Schema;
use crate::search::SegEntry;
use crate::segment::{Backing, DocError, Segment, SegmentBuilder, verify};
use bytes::Bytes;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, PutPayload};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

pub use crate::search::{Hit, SearchRequest, SearchResult, Searcher, SortField};

#[derive(Debug, thiserror::Error)]
pub enum IndexError {
    #[error("storage: {0}")]
    Store(#[from] object_store::Error),
    #[error("bad document: {0}")]
    Doc(#[from] DocError),
    #[error("corrupt file {0}: {1}")]
    Corrupt(String, String),
    #[error("local segment cache: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    pub generation: u64,
    pub segments: Vec<SegmentRef>,
    pub deletes: Vec<DeleteRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRef {
    pub name: String,
    pub generation: u64,
    pub docs: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteRef {
    pub name: String,
    pub generation: u64,
}

enum Op {
    Add(Map<String, Value>),
    Delete(String),
}

/// One atomic unit of change. Operations apply in the order they were added.
#[derive(Default)]
pub struct Batch {
    ops: Vec<Op>,
}

impl Batch {
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, doc: Value) -> Self {
        self.push_add(doc);
        self
    }

    pub fn delete(mut self, id: &str) -> Self {
        self.push_delete(id);
        self
    }

    pub fn push_add(&mut self, doc: Value) {
        match doc {
            Value::Object(m) => self.ops.push(Op::Add(m)),
            // Non-objects fail validation at commit with a clear error.
            other => self
                .ops
                .push(Op::Add(Map::from_iter([("".to_string(), other)]))),
        }
    }

    pub fn push_delete(&mut self, id: &str) {
        self.ops.push(Op::Delete(id.to_string()));
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

pub struct Index {
    store: Arc<dyn ObjectStore>,
    prefix: Path,
    schema: Arc<Schema>,
    /// Highest manifest generation this process has seen; a hint to avoid listing.
    seen_generation: AtomicU64,
    segment_cache: Mutex<HashMap<String, Arc<Segment>>>,
    /// Verified segment files, memory-mapped from here.
    cache_dir: PathBuf,
    /// Owns `cache_dir` when no directory was configured, removing it on drop.
    _cache_tmp: Option<tempfile::TempDir>,
    /// Reused while no newer generation has been committed.
    current: Mutex<Option<Arc<Searcher>>>,
}

fn manifest_name(generation: u64) -> String {
    format!("{generation:020}.json")
}

impl Index {
    /// An index with a private, temporary segment cache.
    pub fn new(store: Arc<dyn ObjectStore>, prefix: &str, schema: Schema) -> Self {
        let tmp = tempfile::Builder::new()
            .prefix("search-segments-")
            .tempdir()
            .expect("create temp dir for segment cache");
        let mut index = Self::with_cache_dir(store, prefix, schema, tmp.path().to_path_buf());
        index._cache_tmp = Some(tmp);
        index
    }

    /// An index caching verified segments in `cache_dir`, which survives
    /// restarts so an instance does not re-download what it already has.
    pub fn with_cache_dir(
        store: Arc<dyn ObjectStore>,
        prefix: &str,
        schema: Schema,
        cache_dir: PathBuf,
    ) -> Self {
        Index {
            store,
            prefix: Path::from(prefix),
            schema: Arc::new(schema),
            seen_generation: AtomicU64::new(0),
            segment_cache: Mutex::default(),
            cache_dir,
            _cache_tmp: None,
            current: Mutex::default(),
        }
    }

    pub fn schema(&self) -> &Schema {
        &self.schema
    }

    fn path(&self, dir: &str, name: &str) -> Path {
        self.prefix.clone().join(dir).join(name)
    }

    async fn read_manifest(&self, generation: u64) -> Result<Option<Manifest>, IndexError> {
        let path = self.path("manifest", &manifest_name(generation));
        match self.store.get(&path).await {
            Ok(r) => {
                let bytes = r.bytes().await?;
                let m = serde_json::from_slice(&bytes)
                    .map_err(|e| IndexError::Corrupt(path.to_string(), e.to_string()))?;
                Ok(Some(m))
            }
            Err(object_store::Error::NotFound { .. }) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    async fn highest_listed_generation(&self) -> Result<u64, IndexError> {
        use futures::TryStreamExt;
        let dir = self.prefix.clone().join("manifest");
        let metas: Vec<_> = self.store.list(Some(&dir)).try_collect().await?;
        Ok(metas
            .iter()
            .filter_map(|m| {
                m.location
                    .filename()?
                    .strip_suffix(".json")?
                    .parse::<u64>()
                    .ok()
            })
            .max()
            .unwrap_or(0))
    }

    /// The newest committed manifest. Probes forward from the last generation
    /// this process saw, listing the directory only on first use.
    pub async fn latest_manifest(&self) -> Result<Manifest, IndexError> {
        let mut generation = self.seen_generation.load(Ordering::Acquire);
        if generation == 0 {
            generation = self.highest_listed_generation().await?;
        }
        let mut current = match generation {
            0 => Manifest::default(),
            g => self.read_manifest(g).await?.ok_or_else(|| {
                IndexError::Corrupt(manifest_name(g), "listed but missing".into())
            })?,
        };
        while let Some(next) = self.read_manifest(current.generation + 1).await? {
            current = next;
        }
        self.seen_generation
            .fetch_max(current.generation, Ordering::AcqRel);
        Ok(current)
    }

    /// Applies a batch atomically and returns the generation it committed at.
    pub async fn commit(&self, batch: Batch) -> Result<u64, IndexError> {
        // Resolve the batch in order: later operations on an id win.
        let mut adds: Vec<(String, Map<String, Value>)> = vec![];
        let mut touched: Vec<String> = vec![];
        for op in batch.ops {
            let id = match &op {
                Op::Add(doc) => match doc.get(&self.schema.unique_key) {
                    Some(Value::String(s)) => s.clone(),
                    Some(Value::Number(n)) => n.to_string(),
                    _ => return Err(DocError::MissingId(self.schema.unique_key.clone()).into()),
                },
                Op::Delete(id) => id.clone(),
            };
            adds.retain(|(existing, _)| *existing != id);
            if let Op::Add(doc) = op {
                adds.push((id.clone(), doc));
            }
            if !touched.contains(&id) {
                touched.push(id);
            }
        }
        let docs: Vec<Map<String, Value>> = adds.into_iter().map(|(_, d)| d).collect();
        let segment = SegmentBuilder::build(&self.schema, &docs)?;

        let create = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        let mut seg_ref = None;
        if segment.doc_count() > 0 {
            let name = format!("{}.seg", ulid::Ulid::generate());
            let payload = PutPayload::from(Bytes::from(segment.encode()));
            self.store
                .put_opts(&self.path("segments", &name), payload, create.clone())
                .await?;
            seg_ref = Some((name, segment.doc_count()));
        }
        let mut del_name = None;
        if !touched.is_empty() {
            let name = format!("{}.json", ulid::Ulid::generate());
            let payload = PutPayload::from(serde_json::to_vec(&touched).expect("ids serialise"));
            self.store
                .put_opts(&self.path("deletes", &name), payload, create.clone())
                .await?;
            del_name = Some(name);
        }

        loop {
            let mut manifest = self.latest_manifest().await?;
            manifest.generation += 1;
            let generation = manifest.generation;
            if let Some((name, docs)) = &seg_ref {
                manifest.segments.push(SegmentRef {
                    name: name.clone(),
                    generation,
                    docs: *docs,
                });
            }
            if let Some(name) = &del_name {
                manifest.deletes.push(DeleteRef {
                    name: name.clone(),
                    generation,
                });
            }
            let body = serde_json::to_vec(&manifest).expect("manifest serialises");
            let path = self.path("manifest", &manifest_name(generation));
            match self
                .store
                .put_opts(&path, PutPayload::from(body), create.clone())
                .await
            {
                Ok(_) => {
                    self.seen_generation.fetch_max(generation, Ordering::AcqRel);
                    return Ok(generation);
                }
                // Another writer won this generation: rebase onto theirs and retry.
                Err(object_store::Error::AlreadyExists { .. }) => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }

    async fn load_segment(&self, name: &str) -> Result<Arc<Segment>, IndexError> {
        if let Some(seg) = self.segment_cache.lock().unwrap().get(name) {
            return Ok(seg.clone());
        }
        let local = self.cache_dir.join(name);
        if !tokio::fs::try_exists(&local).await? {
            let path = self.path("segments", name);
            let bytes = self.store.get(&path).await?.bytes().await?;
            verify(&bytes).map_err(|e| IndexError::Corrupt(path.to_string(), e))?;
            tokio::fs::create_dir_all(&self.cache_dir).await?;
            let tmp = self
                .cache_dir
                .join(format!("{name}.{}.tmp", ulid::Ulid::generate()));
            tokio::fs::write(&tmp, &bytes).await?;
            tokio::fs::rename(&tmp, &local).await?;
        }
        let file = std::fs::File::open(&local)?;
        // SAFETY: cache files are complete before they appear (temp file plus
        // rename) and are never modified afterwards.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let seg = Arc::new(
            Segment::open(Backing::Mmap(mmap))
                .map_err(|e| IndexError::Corrupt(local.display().to_string(), e))?,
        );
        self.segment_cache
            .lock()
            .unwrap()
            .insert(name.to_string(), seg.clone());
        Ok(seg)
    }

    async fn load_deletes(&self, name: &str) -> Result<Vec<String>, IndexError> {
        let path = self.path("deletes", name);
        let bytes = self.store.get(&path).await?.bytes().await?;
        serde_json::from_slice(&bytes)
            .map_err(|e| IndexError::Corrupt(path.to_string(), e.to_string()))
    }

    /// The searcher this process last opened, without touching storage. Request
    /// paths use this; a background task calls [`searcher`](Self::searcher) to
    /// pick up other instances' commits.
    pub async fn cached_searcher(&self) -> Result<Arc<Searcher>, IndexError> {
        let current = self.current.lock().unwrap().clone();
        match current {
            Some(s) => Ok(s),
            None => self.searcher().await,
        }
    }

    /// A point-in-time view of the newest commit.
    pub async fn searcher(&self) -> Result<Arc<Searcher>, IndexError> {
        let manifest = self.latest_manifest().await?;
        let prev = self.current.lock().unwrap().clone();
        if let Some(p) = &prev
            && p.generation() == manifest.generation
        {
            return Ok(p.clone());
        }
        // Manifests only ever append, so everything the previous searcher saw is
        // still present, and only deletes newer than it can change liveness.
        let since = prev.as_ref().map_or(0, |p| p.generation());
        let mut tombstones: HashMap<String, u64> = HashMap::new();
        for d in manifest.deletes.iter().filter(|d| d.generation > since) {
            for id in self.load_deletes(&d.name).await? {
                let g = tombstones.entry(id).or_insert(0);
                *g = (*g).max(d.generation);
            }
        }
        let previous: HashMap<&str, &SegEntry> = prev
            .as_ref()
            .map(|p| p.segments.iter().map(|e| (e.name.as_str(), e)).collect())
            .unwrap_or_default();
        let mut segments = vec![];
        for s in &manifest.segments {
            let (seg, mut live) = match previous.get(s.name.as_str()) {
                Some(e) => (e.seg.clone(), e.live.clone()),
                None => {
                    let seg = self.load_segment(&s.name).await?;
                    let live = vec![true; seg.doc_count() as usize];
                    (seg, live)
                }
            };
            if !tombstones.is_empty() {
                for (doc, alive) in live.iter_mut().enumerate() {
                    if *alive
                        && tombstones
                            .get(seg.ids.get(doc as u32))
                            .is_some_and(|&g| g > s.generation)
                    {
                        *alive = false;
                    }
                }
            }
            segments.push(SegEntry {
                name: s.name.clone(),
                seg,
                live,
            });
        }
        let searcher = Arc::new(Searcher::new(
            manifest.generation,
            self.schema.clone(),
            segments,
        ));
        let mut current = self.current.lock().unwrap();
        if current
            .as_ref()
            .is_none_or(|c| c.generation() < searcher.generation())
        {
            *current = Some(searcher.clone());
        }
        Ok(searcher)
    }
}
