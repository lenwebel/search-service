//! Query execution over a point-in-time set of segments.
//!
//! Scoring is BM25 (k1=1.2, b=0.75) using the Lucene 8+ formula, with term
//! statistics aggregated across all segments so scores do not depend on how
//! documents happen to be split between segments. Multi-term queries
//! (prefix, wildcard, range) are constant-score, as in Lucene.
//!
//! Execution: segments are evaluated in parallel. Each keeps only its own
//! top `start + rows` candidates, which are then merged. Filter (`fq`) results
//! are cached per segment as bitsets; segments never change, so cached
//! filters stay valid across commits and only liveness is applied on top.

use crate::query::{Occur, Query};
use crate::schema::{FieldType, Schema};
use crate::segment::{Num, NumField, Segment, StrField, TextField, parse_numeric};
use rayon::prelude::*;
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::ops::Bound;
use std::sync::Arc;

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[derive(Debug, Clone, PartialEq)]
pub struct SortField {
    pub field: String,
    pub descending: bool,
}

#[derive(Debug, Clone)]
pub struct SearchRequest {
    pub query: Query,
    pub filters: Vec<Query>,
    pub sort: Vec<SortField>,
    pub start: usize,
    pub rows: usize,
}

#[derive(Debug, Clone)]
pub struct Hit {
    pub score: f32,
    pub doc: Map<String, Value>,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub num_found: usize,
    pub max_score: f32,
    pub docs: Vec<Hit>,
}

#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    #[error("can not sort on field '{0}': only string, numeric and date fields are sortable")]
    BadSort(String),
}

/// Matching documents in one segment, sorted by doc number, with scores.
type Matches = Vec<(u32, f32)>;

/// Corpus-wide IDF per field and term, computed once per request.
type Idf = HashMap<String, HashMap<String, f32>>;

pub(crate) struct SegEntry {
    pub name: String,
    pub seg: Arc<Segment>,
    pub live: Vec<bool>,
}

pub struct Searcher {
    generation: u64,
    schema: Arc<Schema>,
    pub(crate) segments: Vec<SegEntry>,
    total_docs: u64,
    /// field -> (docs with field, total tokens), across all segments.
    text_stats: HashMap<String, (u64, u64)>,
}

#[derive(Clone, Copy)]
struct Cand {
    score: f32,
    seg: u32,
    doc: u32,
}

#[derive(Clone, Copy)]
enum Accessor<'a> {
    Score,
    Str(Option<&'a StrField>),
    Num(Option<&'a NumField>),
}

#[derive(PartialEq)]
enum SortKey<'a> {
    Score(f32),
    Str(&'a str),
    Num(Num),
    Missing,
}

impl Accessor<'_> {
    #[inline]
    fn key(&self, c: &Cand) -> SortKey<'_> {
        match self {
            Accessor::Score => SortKey::Score(c.score),
            Accessor::Str(f) => f
                .and_then(|f| f.first(c.doc))
                .map_or(SortKey::Missing, SortKey::Str),
            Accessor::Num(f) => f
                .and_then(|f| f.values(c.doc).next())
                .map_or(SortKey::Missing, SortKey::Num),
        }
    }
}

impl Searcher {
    pub(crate) fn new(generation: u64, schema: Arc<Schema>, segments: Vec<SegEntry>) -> Self {
        let mut text_stats: HashMap<String, (u64, u64)> = HashMap::new();
        let mut total_docs = 0;
        for e in &segments {
            total_docs += e.seg.doc_count() as u64;
            for (field, t) in e.seg.text_fields() {
                let stats = text_stats.entry(field.to_string()).or_default();
                stats.0 += t.docs_with_field as u64;
                stats.1 += t.total_tokens;
            }
        }
        Searcher {
            generation,
            schema,
            segments,
            total_docs,
            text_stats,
        }
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Live documents across all segments.
    pub fn num_docs(&self) -> usize {
        self.segments
            .iter()
            .map(|e| e.live.iter().filter(|&&l| l).count())
            .sum()
    }

    fn ty(&self, field: &str) -> FieldType {
        self.schema
            .field(field)
            .map(|f| f.ty)
            .unwrap_or(FieldType::Ignored)
    }

    fn idf_from(df: u64, n: u64) -> f32 {
        let (df, n) = (df as f32, n.max(df) as f32);
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    fn compute_idf(&self, field: &str, term: &str) -> f32 {
        match self.ty(field) {
            FieldType::Text => {
                let df: u64 = self
                    .segments
                    .iter()
                    .filter_map(|e| {
                        let t = e.seg.text(field)?;
                        Some(t.df(t.terms.find(term)?) as u64)
                    })
                    .sum();
                Self::idf_from(df, self.text_stats.get(field).map_or(0, |s| s.0))
            }
            _ => {
                let df: u64 = self
                    .segments
                    .iter()
                    .filter_map(|e| {
                        let s = e.seg.strings(field)?;
                        Some(s.df(s.terms.find(term)?) as u64)
                    })
                    .sum();
                Self::idf_from(df, self.total_docs)
            }
        }
    }

    /// IDF for every scored term in the query, so segments only look them up.
    fn prepare_idf(&self, q: &Query, out: &mut Idf) {
        let mut add = |field: &str, term: &str| {
            if !out.get(field).is_some_and(|m| m.contains_key(term)) {
                let v = self.compute_idf(field, term);
                out.entry(field.to_string())
                    .or_default()
                    .insert(term.to_string(), v);
            }
        };
        match q {
            Query::Term { field, term }
                if matches!(self.ty(field), FieldType::Text | FieldType::Str) =>
            {
                add(field, term)
            }
            Query::Phrase { field, terms, .. } => terms.iter().for_each(|t| add(field, t)),
            Query::Bool(clauses) => clauses.iter().for_each(|(_, c)| self.prepare_idf(c, out)),
            Query::Boost(inner, _) | Query::Const(inner, _) => self.prepare_idf(inner, out),
            _ => {}
        }
    }

    fn idf(&self, idf: &Idf, field: &str, term: &str) -> f32 {
        idf.get(field)
            .and_then(|m| m.get(term))
            .copied()
            .unwrap_or_else(|| self.compute_idf(field, term))
    }

    #[inline]
    fn bm25_tf(&self, field: &str, freq: f32, len: u32) -> f32 {
        let (docs, tokens) = self.text_stats.get(field).copied().unwrap_or((1, 1));
        let avgdl = tokens as f32 / docs.max(1) as f32;
        freq / (freq + K1 * (1.0 - B + B * len as f32 / avgdl.max(f32::EPSILON)))
    }

    pub fn search(&self, req: &SearchRequest) -> Result<SearchResult, SearchError> {
        self.search_fields(req, None)
    }

    /// Like [`search`](Self::search), returning only the stored fields in `fl`
    /// (`None` or `*` for all). Asking only for the unique key skips
    /// decompressing stored documents entirely.
    pub fn search_fields(
        &self,
        req: &SearchRequest,
        fl: Option<&[&str]>,
    ) -> Result<SearchResult, SearchError> {
        let mut specs: Vec<(FieldType, &str, bool)> = vec![];
        for s in &req.sort {
            let ty = if s.field == "score" {
                FieldType::Ignored
            } else {
                self.ty(&s.field)
            };
            if s.field != "score"
                && !matches!(
                    ty,
                    FieldType::Str | FieldType::Long | FieldType::Double | FieldType::Date
                )
            {
                return Err(SearchError::BadSort(s.field.clone()));
            }
            specs.push((ty, s.field.as_str(), s.descending));
        }
        if specs.is_empty() {
            specs.push((FieldType::Ignored, "score", true));
        }
        let accessors: Vec<Vec<Accessor<'_>>> = self
            .segments
            .iter()
            .map(|e| {
                specs
                    .iter()
                    .map(|(ty, field, _)| match (ty, *field) {
                        (_, "score") => Accessor::Score,
                        (FieldType::Str, f) => Accessor::Str(e.seg.strings(f)),
                        (_, f) => Accessor::Num(e.seg.numbers(f)),
                    })
                    .collect()
            })
            .collect();
        let cmp = |a: &Cand, b: &Cand| -> Ordering {
            let (aa, ba) = (&accessors[a.seg as usize], &accessors[b.seg as usize]);
            for (i, (_, _, desc)) in specs.iter().enumerate() {
                let o = compare_keys(&aa[i].key(a), &ba[i].key(b), *desc);
                if o != Ordering::Equal {
                    return o;
                }
            }
            (a.seg, a.doc).cmp(&(b.seg, b.doc))
        };

        let mut idf = Idf::new();
        self.prepare_idf(&req.query, &mut idf);
        let filter_keys: Vec<String> = req.filters.iter().map(|f| format!("{f:?}")).collect();
        let k = req.start + req.rows;

        let per_segment: Vec<(usize, Vec<Cand>, f32)> = self
            .segments
            .par_iter()
            .enumerate()
            .map(|(si, e)| {
                let n = e.seg.doc_count();
                let filters: Vec<Arc<Vec<u64>>> = req
                    .filters
                    .iter()
                    .zip(&filter_keys)
                    .map(|(f, key)| {
                        e.seg
                            .cached_filter(key, || to_bits(n, &self.eval(&idf, &e.seg, f)))
                    })
                    .collect();
                let mut m: Matches = match (&req.query, filters.first()) {
                    // Browsing a filtered set: walk the cached bitset directly.
                    (Query::MatchAll, Some(bits)) => iter_bits(bits).map(|d| (d, 1.0)).collect(),
                    _ => self.eval(&idf, &e.seg, &req.query),
                };
                m.retain(|&(d, _)| e.live[d as usize] && filters.iter().all(|b| test_bit(b, d)));
                let count = m.len();
                let max = m.iter().map(|&(_, s)| s).fold(0.0, f32::max);
                if k == 0 {
                    return (count, vec![], max);
                }
                let mut cands: Vec<Cand> = m
                    .into_iter()
                    .map(|(doc, score)| Cand {
                        score,
                        seg: si as u32,
                        doc,
                    })
                    .collect();
                top_k(&mut cands, k, cmp);
                (count, cands, max)
            })
            .collect();

        let num_found = per_segment.iter().map(|p| p.0).sum();
        let max_score = per_segment.iter().map(|p| p.2).fold(0.0, f32::max);
        let mut cands: Vec<Cand> = per_segment.into_iter().flat_map(|p| p.1).collect();
        top_k(&mut cands, k, cmp);
        cands.sort_by(cmp);
        let page: Vec<Cand> = cands.into_iter().skip(req.start).take(req.rows).collect();
        let docs = self.fetch(&page, fl);
        Ok(SearchResult {
            num_found,
            max_score,
            docs,
        })
    }

    /// Loads stored fields for a page of hits, decompressing each block once.
    fn fetch(&self, page: &[Cand], fl: Option<&[&str]>) -> Vec<Hit> {
        let fl = fl.filter(|f| !f.contains(&"*"));
        let key = self.schema.unique_key.as_str();
        let id_only = fl.is_some_and(|f| f.iter().all(|n| *n == key || *n == "score"));
        let mut by_seg: HashMap<u32, Vec<u32>> = HashMap::new();
        if !id_only {
            page.iter()
                .for_each(|c| by_seg.entry(c.seg).or_default().push(c.doc));
        }
        let mut loaded: HashMap<(u32, u32), Map<String, Value>> = HashMap::new();
        for (si, docs) in by_seg {
            let seg = &self.segments[si as usize].seg;
            for (doc, stored) in docs.iter().zip(seg.stored_many(&docs)) {
                loaded.insert((si, *doc), stored);
            }
        }
        page.iter()
            .map(|c| {
                let doc = if id_only {
                    let mut m = Map::new();
                    if fl.is_some_and(|f| f.contains(&key)) {
                        m.insert(
                            key.to_string(),
                            Value::String(
                                self.segments[c.seg as usize].seg.ids.get(c.doc).to_string(),
                            ),
                        );
                    }
                    m
                } else {
                    let full = loaded.remove(&(c.seg, c.doc)).unwrap_or_default();
                    match fl {
                        None => full,
                        Some(f) => full
                            .into_iter()
                            .filter(|(k, _)| f.contains(&k.as_str()))
                            .collect(),
                    }
                };
                Hit {
                    score: c.score,
                    doc,
                }
            })
            .collect()
    }

    fn eval(&self, idf: &Idf, seg: &Segment, q: &Query) -> Matches {
        match q {
            Query::MatchAll => (0..seg.doc_count()).map(|d| (d, 1.0)).collect(),
            Query::MatchNone => vec![],
            Query::Term { field, term } => self.eval_term(idf, seg, field, term),
            Query::Phrase { field, terms, slop } => self.eval_phrase(idf, seg, field, terms, *slop),
            Query::Prefix { field, prefix } => self.eval_terms_matching(
                seg,
                field,
                Bound::Included(prefix.as_str()),
                |t| !t.starts_with(prefix.as_str()),
                |_| true,
            ),
            Query::Wildcard { field, pattern } => self.eval_wildcard(seg, field, pattern),
            Query::Range {
                field,
                lower,
                upper,
            } => self.eval_range(seg, field, lower, upper),
            Query::Boost(inner, b) => {
                let mut m = self.eval(idf, seg, inner);
                m.iter_mut().for_each(|(_, s)| *s *= b);
                m
            }
            Query::Const(inner, c) => {
                let mut m = self.eval(idf, seg, inner);
                m.iter_mut().for_each(|(_, s)| *s = *c);
                m
            }
            Query::Bool(clauses) => self.eval_bool(idf, seg, clauses),
        }
    }

    fn eval_bool(&self, idf: &Idf, seg: &Segment, clauses: &[(Occur, Query)]) -> Matches {
        let mut must: Option<Matches> = None;
        let mut should: Option<Matches> = None;
        let mut must_not: Matches = vec![];
        for (occur, q) in clauses {
            if *occur == Occur::Must && must.as_ref().is_some_and(Vec::is_empty) {
                continue; // already impossible
            }
            let m = self.eval(idf, seg, q);
            match occur {
                Occur::Must => {
                    must = Some(match must {
                        None => m,
                        Some(acc) => intersect(&acc, &m),
                    })
                }
                Occur::Should => {
                    should = Some(match should {
                        None => m,
                        Some(acc) => union(&acc, &m),
                    })
                }
                Occur::MustNot => {
                    must_not = if must_not.is_empty() {
                        m
                    } else {
                        union(&must_not, &m)
                    }
                }
            }
        }
        let base = match (must, should) {
            (Some(must), Some(should)) => add_optional(must, &should),
            (Some(must), None) => must,
            (None, Some(should)) => should,
            // Purely negative: match everything else, as Solr does.
            (None, None) => (0..seg.doc_count()).map(|d| (d, 1.0)).collect(),
        };
        if must_not.is_empty() {
            base
        } else {
            difference(base, &must_not)
        }
    }

    fn eval_term(&self, idf: &Idf, seg: &Segment, field: &str, term: &str) -> Matches {
        match self.ty(field) {
            FieldType::Text => {
                let Some(tf) = seg.text(field) else {
                    return vec![];
                };
                let Some(ord) = tf.terms.find(term) else {
                    return vec![];
                };
                let idf = self.idf(idf, field, term);
                let (docs, tokens) = self.text_stats.get(field).copied().unwrap_or((1, 1));
                let avgdl = (tokens as f32 / docs.max(1) as f32).max(f32::EPSILON);
                let mut postings = tf.postings(ord);
                let mut positions = vec![];
                let mut out = Vec::with_capacity(tf.df(ord) as usize);
                while let Some(doc) = postings.next(&mut positions) {
                    let freq = positions.len() as f32;
                    let norm = K1 * (1.0 - B + B * tf.length(doc) as f32 / avgdl);
                    out.push((doc, idf * freq / (freq + norm)));
                }
                out
            }
            FieldType::Str => {
                let Some(sf) = seg.strings(field) else {
                    return vec![];
                };
                let Some(ord) = sf.terms.find(term) else {
                    return vec![];
                };
                let idf = self.idf(idf, field, term);
                let mut out = Vec::with_capacity(sf.df(ord) as usize);
                out.extend(sf.docs(ord).map(|d| (d, idf)));
                out
            }
            ty @ (FieldType::Long | FieldType::Double | FieldType::Date) => {
                let Some(target) = parse_numeric(ty, term) else {
                    return vec![];
                };
                self.scan_numbers(seg, field, |n| n.compare(&target) == Ordering::Equal)
            }
            FieldType::Ignored => vec![],
        }
    }

    fn scan_numbers(&self, seg: &Segment, field: &str, pred: impl Fn(&Num) -> bool) -> Matches {
        let Some(col) = seg.numbers(field) else {
            return vec![];
        };
        (0..seg.doc_count())
            .filter(|&d| col.values(d).any(|v| pred(&v)))
            .map(|d| (d, 1.0))
            .collect()
    }

    fn eval_phrase(
        &self,
        idf: &Idf,
        seg: &Segment,
        field: &str,
        terms: &[String],
        slop: u32,
    ) -> Matches {
        let Some(tf) = seg.text(field) else {
            return vec![];
        };
        let ords: Option<Vec<u32>> = terms.iter().map(|t| tf.terms.find(t)).collect();
        let Some(ords) = ords else { return vec![] };
        let lists: Vec<Decoded> = ords.iter().map(|&o| Decoded::from(tf, o)).collect();
        let idf: f32 = terms.iter().map(|t| self.idf(idf, field, t)).sum();
        let mut cursors = vec![0usize; lists.len()];
        let mut at = Vec::with_capacity(lists.len());
        let mut out = vec![];
        'docs: for d0 in 0..lists[0].docs.len() {
            let doc = lists[0].docs[d0].0;
            at.clear();
            at.push(d0);
            for (i, list) in lists.iter().enumerate().skip(1) {
                while cursors[i] < list.docs.len() && list.docs[cursors[i]].0 < doc {
                    cursors[i] += 1;
                }
                match list.docs.get(cursors[i]) {
                    Some(&(d, _, _)) if d == doc => at.push(cursors[i]),
                    _ => continue 'docs,
                }
            }
            // A match starts at each position of the first term from which every
            // later term sits within `slop` of its expected offset.
            let freq = lists[0]
                .positions(at[0])
                .iter()
                .filter(|&&start| {
                    (1..lists.len()).all(|i| {
                        let expected = start as i64 + i as i64;
                        lists[i]
                            .positions(at[i])
                            .iter()
                            .any(|&p| (p as i64 - expected).unsigned_abs() <= slop as u64)
                    })
                })
                .count();
            if freq > 0 {
                out.push((doc, idf * self.bm25_tf(field, freq as f32, tf.length(doc))));
            }
        }
        out
    }

    /// Constant-score union over dictionary terms, walking forward from `from`
    /// until `stop` holds and including the terms for which `keep` holds.
    fn eval_terms_matching(
        &self,
        seg: &Segment,
        field: &str,
        from: Bound<&str>,
        stop: impl Fn(&str) -> bool,
        keep: impl Fn(&str) -> bool,
    ) -> Matches {
        let n = seg.doc_count();
        let mut bits = vec![0u64; (n as usize).div_ceil(64)];
        let mut any = false;
        match self.ty(field) {
            FieldType::Text => {
                if let Some(tf) = seg.text(field) {
                    let mut positions = vec![];
                    for ord in tf.terms.lower_bound(from)..tf.terms.len() {
                        let t = tf.terms.get(ord);
                        if stop(t) {
                            break;
                        }
                        if !keep(t) {
                            continue;
                        }
                        let mut p = tf.postings(ord);
                        while let Some(d) = p.next(&mut positions) {
                            set_bit(&mut bits, d);
                            any = true;
                        }
                    }
                }
            }
            FieldType::Str => {
                if let Some(sf) = seg.strings(field) {
                    for ord in sf.terms.lower_bound(from)..sf.terms.len() {
                        let t = sf.terms.get(ord);
                        if stop(t) {
                            break;
                        }
                        if !keep(t) {
                            continue;
                        }
                        for d in sf.docs(ord) {
                            set_bit(&mut bits, d);
                            any = true;
                        }
                    }
                }
            }
            _ => {}
        }
        if !any {
            return vec![];
        }
        iter_bits(&bits).map(|d| (d, 1.0)).collect()
    }

    fn eval_wildcard(&self, seg: &Segment, field: &str, pattern: &str) -> Matches {
        if pattern == "*" {
            // Field exists.
            let n = seg.doc_count();
            return match self.ty(field) {
                FieldType::Text => seg.text(field).map_or(vec![], |f| {
                    (0..n)
                        .filter(|&d| f.length(d) > 0)
                        .map(|d| (d, 1.0))
                        .collect()
                }),
                FieldType::Str => seg.strings(field).map_or(vec![], |f| {
                    (0..n)
                        .filter(|&d| f.first(d).is_some())
                        .map(|d| (d, 1.0))
                        .collect()
                }),
                FieldType::Ignored => vec![],
                _ => self.scan_numbers(seg, field, |_| true),
            };
        }
        let literal_prefix = &pattern[..pattern.find(['*', '?']).unwrap_or(pattern.len())];
        self.eval_terms_matching(
            seg,
            field,
            Bound::Included(literal_prefix),
            |t| !t.starts_with(literal_prefix),
            |t| glob(pattern, t),
        )
    }

    fn eval_range(
        &self,
        seg: &Segment,
        field: &str,
        lower: &Bound<String>,
        upper: &Bound<String>,
    ) -> Matches {
        let ty = self.ty(field);
        if ty.is_numeric() {
            let parse = |b: &Bound<String>| -> Option<Bound<Num>> {
                Some(match b {
                    Bound::Included(s) => Bound::Included(parse_numeric(ty, s)?),
                    Bound::Excluded(s) => Bound::Excluded(parse_numeric(ty, s)?),
                    Bound::Unbounded => Bound::Unbounded,
                })
            };
            let (Some(lo), Some(hi)) = (parse(lower), parse(upper)) else {
                return vec![];
            };
            return self.scan_numbers(seg, field, |n| {
                let above = match &lo {
                    Bound::Included(l) => n.compare(l) != Ordering::Less,
                    Bound::Excluded(l) => n.compare(l) == Ordering::Greater,
                    Bound::Unbounded => true,
                };
                let below = match &hi {
                    Bound::Included(h) => n.compare(h) != Ordering::Greater,
                    Bound::Excluded(h) => n.compare(h) == Ordering::Less,
                    Bound::Unbounded => true,
                };
                above && below
            });
        }
        let from = match lower {
            Bound::Included(s) => Bound::Included(s.as_str()),
            Bound::Excluded(s) => Bound::Excluded(s.as_str()),
            Bound::Unbounded => Bound::Unbounded,
        };
        let below = |t: &str| match upper {
            Bound::Included(h) => t <= h.as_str(),
            Bound::Excluded(h) => t < h.as_str(),
            Bound::Unbounded => true,
        };
        self.eval_terms_matching(seg, field, from, |t| !below(t), |_| true)
    }
}

/// Keeps the `k` best candidates (unordered) in linear time.
fn top_k(v: &mut Vec<Cand>, k: usize, cmp: impl Fn(&Cand, &Cand) -> Ordering) {
    if k == 0 {
        v.clear();
    } else if v.len() > k {
        v.select_nth_unstable_by(k - 1, cmp);
        v.truncate(k);
    }
}

fn to_bits(n: u32, m: &Matches) -> Vec<u64> {
    let mut bits = vec![0u64; (n as usize).div_ceil(64)];
    m.iter().for_each(|&(d, _)| set_bit(&mut bits, d));
    bits
}

#[inline]
fn set_bit(bits: &mut [u64], d: u32) {
    bits[(d >> 6) as usize] |= 1 << (d & 63);
}

#[inline]
fn test_bit(bits: &[u64], d: u32) -> bool {
    bits[(d >> 6) as usize] & (1 << (d & 63)) != 0
}

fn iter_bits(bits: &[u64]) -> impl Iterator<Item = u32> + '_ {
    bits.iter().enumerate().flat_map(|(w, &word)| {
        let mut word = word;
        std::iter::from_fn(move || {
            if word == 0 {
                return None;
            }
            let t = word.trailing_zeros();
            word &= word - 1;
            Some((w as u32) * 64 + t)
        })
    })
}

/// One term's postings decoded for phrase matching: (doc, start, end) into `pos`.
struct Decoded {
    docs: Vec<(u32, u32, u32)>,
    pos: Vec<u32>,
}

impl Decoded {
    fn from(tf: &TextField, ord: u32) -> Decoded {
        let mut d = Decoded {
            docs: Vec::with_capacity(tf.df(ord) as usize),
            pos: vec![],
        };
        let mut it = tf.postings(ord);
        let mut buf = vec![];
        while let Some(doc) = it.next(&mut buf) {
            let start = d.pos.len() as u32;
            d.pos.extend_from_slice(&buf);
            d.docs.push((doc, start, d.pos.len() as u32));
        }
        d
    }

    fn positions(&self, i: usize) -> &[u32] {
        let (_, s, e) = self.docs[i];
        &self.pos[s as usize..e as usize]
    }
}

fn compare_keys(a: &SortKey<'_>, b: &SortKey<'_>, desc: bool) -> Ordering {
    let o = match (a, b) {
        // Missing values sort last in either direction (Solr's sortMissingLast).
        (SortKey::Missing, SortKey::Missing) => return Ordering::Equal,
        (SortKey::Missing, _) => return Ordering::Greater,
        (_, SortKey::Missing) => return Ordering::Less,
        (SortKey::Score(x), SortKey::Score(y)) => x.partial_cmp(y).unwrap_or(Ordering::Equal),
        (SortKey::Str(x), SortKey::Str(y)) => x.cmp(y),
        (SortKey::Num(x), SortKey::Num(y)) => x.compare(y),
        _ => Ordering::Equal,
    };
    if desc { o.reverse() } else { o }
}

/// `*` matches any run of characters, `?` exactly one.
fn glob(pattern: &str, text: &str) -> bool {
    if !pattern.contains('?') {
        // Common case: literal pieces separated by `*`, matched without allocating.
        let mut pieces = pattern.split('*');
        let first = pieces.next().unwrap_or("");
        let Some(mut rest) = text.strip_prefix(first) else {
            return false;
        };
        let pieces: Vec<&str> = pieces.collect();
        let Some((last, middle)) = pieces.split_last() else {
            return rest.is_empty();
        };
        for piece in middle {
            match rest.find(piece) {
                Some(i) => rest = &rest[i + piece.len()..],
                None => return false,
            }
        }
        return rest.ends_with(last);
    }
    let (p, t): (Vec<char>, Vec<char>) = (pattern.chars().collect(), text.chars().collect());
    let (mut pi, mut ti, mut star, mut mark) = (0, 0, None, 0);
    while ti < t.len() {
        if pi < p.len() && (p[pi] == '?' || p[pi] == t[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < p.len() && p[pi] == '*' {
            star = Some(pi);
            mark = ti;
            pi += 1;
        } else if let Some(s) = star {
            pi = s + 1;
            mark += 1;
            ti = mark;
        } else {
            return false;
        }
    }
    p[pi..].iter().all(|&c| c == '*')
}

fn intersect(a: &Matches, b: &Matches) -> Matches {
    let (mut i, mut j, mut out) = (0, 0, Vec::with_capacity(a.len().min(b.len())));
    while i < a.len() && j < b.len() {
        match a[i].0.cmp(&b[j].0) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out.push((a[i].0, a[i].1 + b[j].1));
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn union(a: &Matches, b: &Matches) -> Matches {
    let (mut i, mut j, mut out) = (0, 0, Vec::with_capacity(a.len() + b.len()));
    while i < a.len() || j < b.len() {
        if j >= b.len() || (i < a.len() && a[i].0 < b[j].0) {
            out.push(a[i]);
            i += 1;
        } else if i >= a.len() || b[j].0 < a[i].0 {
            out.push(b[j]);
            j += 1;
        } else {
            out.push((a[i].0, a[i].1 + b[j].1));
            i += 1;
            j += 1;
        }
    }
    out
}

/// Required matches, plus the score of any optional match on the same document.
fn add_optional(mut required: Matches, optional: &Matches) -> Matches {
    let mut j = 0;
    for (d, s) in required.iter_mut() {
        while j < optional.len() && optional[j].0 < *d {
            j += 1;
        }
        if let Some(&(od, os)) = optional.get(j)
            && od == *d
        {
            *s += os;
        }
    }
    required
}

fn difference(mut a: Matches, b: &Matches) -> Matches {
    let mut j = 0;
    a.retain(|(d, _)| {
        while j < b.len() && b[j].0 < *d {
            j += 1;
        }
        !(j < b.len() && b[j].0 == *d)
    });
    a
}
