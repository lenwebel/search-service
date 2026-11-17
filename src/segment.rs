//! Immutable segments: built in memory by one writer, encoded once into a
//! compact binary file, and read back through a memory map.
//!
//! File layout (v2), all integers little-endian:
//!
//! ```text
//! header   magic "SSG2" | version u32 | doc_count u32 | directory offset u64
//! sections ids, stored documents, one section per field (any order)
//! directory  ids offset u64 | stored offset u64 | field count u32 |
//!            per field: name len u16, name, kind u8, section offset u64
//! trailer  crc32 of every preceding byte
//! ```
//!
//! - String tables (ids, term dictionaries): count u32, (count+1) u32 end
//!   offsets, then the concatenated UTF-8 bytes. Terms are sorted, so lookup
//!   is a binary search and prefix/range scans walk forward.
//! - Text postings: per document, varint doc delta, varint freq, then freq
//!   varint position deltas.
//! - String postings: varint doc deltas. Plus each document's first value as
//!   a term ordinal, for sorting.
//! - Numbers: (doc_count+1) u32 value offsets, then 8-byte values.
//! - Stored documents: zstd-compressed blocks of `STORED_BLOCK` documents.

use crate::analysis::tokenize;
use crate::schema::{FieldType, Schema};
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::ops::{Bound, Deref};
use std::sync::Arc;

const MAGIC: &[u8; 4] = b"SSG2";
pub const FORMAT_VERSION: u32 = 2;
const STORED_BLOCK: usize = 64;
const NO_ORD: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Num {
    I(i64),
    F(f64),
}

impl Num {
    pub fn compare(&self, other: &Num) -> Ordering {
        match (self, other) {
            (Num::I(a), Num::I(b)) => a.cmp(b),
            (a, b) => a
                .as_f64()
                .partial_cmp(&b.as_f64())
                .unwrap_or(Ordering::Equal),
        }
    }

    fn as_f64(&self) -> f64 {
        match *self {
            Num::I(i) => i as f64,
            Num::F(f) => f,
        }
    }
}

/// Parses a query or document value for a numeric field.
pub fn parse_numeric(ty: FieldType, text: &str) -> Option<Num> {
    let text = text.trim();
    match ty {
        FieldType::Long => text.parse::<i64>().ok().map(Num::I),
        FieldType::Double => text.parse::<f64>().ok().map(Num::F),
        FieldType::Date => chrono::DateTime::parse_from_rfc3339(text)
            .ok()
            .map(|d| Num::I(d.timestamp_millis())),
        _ => None,
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DocError {
    #[error("document is missing unique key field '{0}'")]
    MissingId(String),
    #[error("unknown field '{0}'")]
    UnknownField(String),
    #[error("invalid value {1} for field '{0}'")]
    BadValue(String, String),
}

// ---------------------------------------------------------------- building

/// Solr's default positionIncrementGap for text_general: stops phrases
/// matching across two values of a multi-valued field.
const POSITION_GAP: u32 = 100;

#[derive(Default)]
struct MemText {
    terms: BTreeMap<String, Vec<(u32, Vec<u32>)>>,
    lengths: Vec<u32>,
}

#[derive(Default)]
struct MemStr {
    terms: BTreeMap<String, Vec<u32>>,
    first: Vec<Option<String>>,
}

/// A segment under construction. Transient: lives for one commit.
pub struct SegmentBuilder {
    ids: Vec<String>,
    stored: Vec<String>,
    text: BTreeMap<String, MemText>,
    strings: BTreeMap<String, MemStr>,
    numbers: BTreeMap<String, (bool, Vec<Vec<Num>>)>,
}

impl SegmentBuilder {
    pub fn doc_count(&self) -> u32 {
        self.ids.len() as u32
    }

    pub fn build(schema: &Schema, docs: &[Map<String, Value>]) -> Result<SegmentBuilder, DocError> {
        let mut seg = SegmentBuilder {
            ids: vec![],
            stored: vec![],
            text: BTreeMap::new(),
            strings: BTreeMap::new(),
            numbers: BTreeMap::new(),
        };
        let n = docs.len();
        for (doc_no, doc) in docs.iter().enumerate() {
            let id = match doc.get(&schema.unique_key) {
                Some(Value::String(s)) => s.clone(),
                Some(Value::Number(n)) => n.to_string(),
                _ => return Err(DocError::MissingId(schema.unique_key.clone())),
            };
            let mut next_pos: HashMap<String, u32> = HashMap::new();
            for (name, value) in doc {
                let values: Vec<&Value> = match value {
                    Value::Array(a) => a.iter().collect(),
                    Value::Null => continue,
                    v => vec![v],
                };
                let mut targets = vec![name.clone()];
                targets.extend(schema.copy_targets(name));
                for target in targets {
                    let info = schema
                        .field(&target)
                        .ok_or_else(|| DocError::UnknownField(target.clone()))?;
                    for v in &values {
                        seg.index_value(&target, info.ty, v, doc_no as u32, n, &mut next_pos)?;
                    }
                }
            }
            seg.ids.push(id);
            seg.stored.push(Value::Object(doc.clone()).to_string());
        }
        Ok(seg)
    }

    fn index_value(
        &mut self,
        field: &str,
        ty: FieldType,
        value: &Value,
        doc: u32,
        n_docs: usize,
        next_pos: &mut HashMap<String, u32>,
    ) -> Result<(), DocError> {
        let text = match value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            other => return Err(DocError::BadValue(field.into(), other.to_string())),
        };
        match ty {
            FieldType::Text => {
                let tf = self.text.entry(field.into()).or_default();
                tf.lengths.resize(n_docs, 0);
                let start = next_pos.entry(field.into()).or_insert(0);
                let tokens = tokenize(&text);
                for (i, tok) in tokens.iter().enumerate() {
                    let pos = *start + i as u32;
                    let list = tf.terms.entry(tok.clone()).or_default();
                    match list.last_mut() {
                        Some((d, positions)) if *d == doc => positions.push(pos),
                        _ => list.push((doc, vec![pos])),
                    }
                }
                tf.lengths[doc as usize] += tokens.len() as u32;
                *start += tokens.len() as u32 + POSITION_GAP;
            }
            FieldType::Str => {
                let sf = self.strings.entry(field.into()).or_default();
                sf.first.resize(n_docs, None);
                let list = sf.terms.entry(text.clone()).or_default();
                if list.last() != Some(&doc) {
                    list.push(doc);
                }
                sf.first[doc as usize].get_or_insert(text);
            }
            FieldType::Long | FieldType::Double | FieldType::Date => {
                let num = match (ty, value) {
                    (FieldType::Long, Value::Number(n)) => n.as_i64().map(Num::I),
                    (FieldType::Double, Value::Number(n)) => n.as_f64().map(Num::F),
                    _ => parse_numeric(ty, &text),
                }
                .ok_or_else(|| DocError::BadValue(field.into(), value.to_string()))?;
                let col = self
                    .numbers
                    .entry(field.into())
                    .or_insert((ty == FieldType::Double, vec![]));
                col.1.resize(n_docs, vec![]);
                col.1[doc as usize].push(num);
            }
            FieldType::Ignored => {}
        }
        Ok(())
    }

    pub fn encode(&self) -> Vec<u8> {
        let n = self.ids.len();
        let mut w = Writer(Vec::with_capacity(1 << 16));
        w.0.extend_from_slice(MAGIC);
        w.u32(FORMAT_VERSION);
        w.u32(n as u32);
        let dir_slot = w.0.len();
        w.u64(0);

        let ids_off = w.str_table(self.ids.iter().map(String::as_str));

        let stored_off = w.pos();
        let blocks: Vec<&[String]> = self.stored.chunks(STORED_BLOCK).collect();
        w.u32(STORED_BLOCK as u32);
        w.u32(blocks.len() as u32);
        let frames: Vec<Vec<u8>> = blocks
            .iter()
            .map(|block| {
                let mut raw = vec![];
                for doc in *block {
                    raw.extend_from_slice(&(doc.len() as u32).to_le_bytes());
                    raw.extend_from_slice(doc.as_bytes());
                }
                zstd::bulk::compress(&raw, 3).expect("zstd compression of an in-memory buffer")
            })
            .collect();
        let mut end = 0u64;
        w.u64(0);
        for f in &frames {
            end += f.len() as u64;
            w.u64(end);
        }
        frames.iter().for_each(|f| w.0.extend_from_slice(f));

        let mut fields: Vec<(String, u8, u64)> = vec![];
        for (name, tf) in &self.text {
            let mut lengths = tf.lengths.clone();
            lengths.resize(n, 0);
            let mut postings = vec![];
            let mut ends = vec![];
            for list in tf.terms.values() {
                let mut prev = 0u32;
                for (doc, positions) in list {
                    put_varint(&mut postings, doc - prev);
                    prev = *doc;
                    put_varint(&mut postings, positions.len() as u32);
                    let mut last = 0u32;
                    for &p in positions {
                        put_varint(&mut postings, p - last);
                        last = p;
                    }
                }
                ends.push(postings.len() as u32);
            }
            let header = w.pos();
            let docs_with_field = lengths.iter().filter(|&&l| l > 0).count() as u32;
            let total_tokens: u64 = lengths.iter().map(|&l| l as u64).sum();
            w.u32(docs_with_field);
            w.u64(total_tokens);
            let slots = w.reserve(5); // lengths, terms, df, postings ends, postings
            let lengths_off = w.pos();
            lengths.iter().for_each(|&l| w.u32(l));
            let terms_off = w.str_table(tf.terms.keys().map(String::as_str));
            let df_off = w.pos();
            tf.terms.values().for_each(|l| w.u32(l.len() as u32));
            let ends_off = w.pos();
            w.u32(0);
            ends.iter().for_each(|&e| w.u32(e));
            let post_off = w.pos();
            w.0.extend_from_slice(&postings);
            w.fill(slots, &[lengths_off, terms_off, df_off, ends_off, post_off]);
            fields.push((name.clone(), 0, header));
        }
        for (name, sf) in &self.strings {
            let ords: HashMap<&str, u32> = sf
                .terms
                .keys()
                .enumerate()
                .map(|(i, t)| (t.as_str(), i as u32))
                .collect();
            let mut postings = vec![];
            let mut ends = vec![];
            for docs in sf.terms.values() {
                let mut prev = 0u32;
                for &d in docs {
                    put_varint(&mut postings, d - prev);
                    prev = d;
                }
                ends.push(postings.len() as u32);
            }
            let header = w.pos();
            let slots = w.reserve(5); // terms, df, postings ends, postings, first ords
            let terms_off = w.str_table(sf.terms.keys().map(String::as_str));
            let df_off = w.pos();
            sf.terms.values().for_each(|l| w.u32(l.len() as u32));
            let ends_off = w.pos();
            w.u32(0);
            ends.iter().for_each(|&e| w.u32(e));
            let post_off = w.pos();
            w.0.extend_from_slice(&postings);
            let first_off = w.pos();
            for d in 0..n {
                let ord = sf
                    .first
                    .get(d)
                    .and_then(|v| v.as_deref())
                    .map_or(NO_ORD, |t| ords[t]);
                w.u32(ord);
            }
            w.fill(slots, &[terms_off, df_off, ends_off, post_off, first_off]);
            fields.push((name.clone(), 1, header));
        }
        for (name, (is_float, col)) in &self.numbers {
            let header = w.pos();
            let slots = w.reserve(2);
            let offs_off = w.pos();
            let mut count = 0u32;
            w.u32(0);
            for d in 0..n {
                count += col.get(d).map_or(0, |v| v.len() as u32);
                w.u32(count);
            }
            let vals_off = w.pos();
            for v in col.iter().flatten() {
                match v {
                    Num::I(i) => w.0.extend_from_slice(&i.to_le_bytes()),
                    Num::F(f) => w.0.extend_from_slice(&f.to_le_bytes()),
                }
            }
            w.fill(slots, &[offs_off, vals_off]);
            fields.push((name.clone(), if *is_float { 3 } else { 2 }, header));
        }

        let dir_off = w.pos();
        w.u64(ids_off);
        w.u64(stored_off);
        w.u32(fields.len() as u32);
        for (name, kind, off) in &fields {
            w.0.extend_from_slice(&(name.len() as u16).to_le_bytes());
            w.0.extend_from_slice(name.as_bytes());
            w.0.push(*kind);
            w.u64(*off);
        }
        w.0[dir_slot..dir_slot + 8].copy_from_slice(&dir_off.to_le_bytes());
        let crc = crc32fast::hash(&w.0);
        w.u32(crc);
        w.0
    }
}

struct Writer(Vec<u8>);

impl Writer {
    fn pos(&self) -> u64 {
        self.0.len() as u64
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn reserve(&mut self, slots: usize) -> usize {
        let at = self.0.len();
        self.0.resize(at + slots * 8, 0);
        at
    }
    fn fill(&mut self, at: usize, values: &[u64]) {
        for (i, v) in values.iter().enumerate() {
            self.0[at + i * 8..at + i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
    fn str_table<'a>(&mut self, items: impl Iterator<Item = &'a str>) -> u64 {
        let items: Vec<&str> = items.collect();
        let off = self.pos();
        self.u32(items.len() as u32);
        let mut end = 0u32;
        self.u32(0);
        for s in &items {
            end += s.len() as u32;
            self.u32(end);
        }
        items
            .iter()
            .for_each(|s| self.0.extend_from_slice(s.as_bytes()));
        off
    }
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

#[inline]
fn get_varint(buf: &[u8], pos: &mut usize) -> u32 {
    let mut v = 0u32;
    let mut shift = 0;
    loop {
        let b = buf[*pos];
        *pos += 1;
        v |= ((b & 0x7f) as u32) << shift;
        if b < 0x80 {
            return v;
        }
        shift += 7;
    }
}

// ---------------------------------------------------------------- reading

/// Segment bytes: a memory map of a cached file, or an owned buffer.
pub enum Backing {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Deref for Backing {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Backing::Mmap(m) => m,
            Backing::Owned(v) => v,
        }
    }
}

/// Checks magic, version and the trailing CRC32. Run once, when a segment is
/// downloaded, before it is cached or served.
pub fn verify(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < 24 || &bytes[..4] != MAGIC {
        return Err("not a segment file".into());
    }
    let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
    if version != FORMAT_VERSION {
        return Err(format!("unsupported segment format {version}"));
    }
    let (body, trailer) = bytes.split_at(bytes.len() - 4);
    let expected = u32::from_le_bytes(trailer.try_into().unwrap());
    if crc32fast::hash(body) != expected {
        return Err("checksum mismatch".into());
    }
    Ok(())
}

#[derive(Clone)]
struct Bytes(Arc<Backing>);

impl Bytes {
    #[inline]
    fn u32(&self, at: usize) -> u32 {
        u32::from_le_bytes(self.0[at..at + 4].try_into().unwrap())
    }
    #[inline]
    fn u64(&self, at: usize) -> u64 {
        u64::from_le_bytes(self.0[at..at + 8].try_into().unwrap())
    }
    #[inline]
    fn usize(&self, at: usize) -> usize {
        self.u64(at) as usize
    }
}

/// A sorted (or, for ids, insertion-ordered) table of strings.
#[derive(Clone)]
pub struct StrTable {
    b: Bytes,
    count: u32,
    ends: usize,
    data: usize,
}

impl StrTable {
    fn open(b: &Bytes, at: usize) -> StrTable {
        let count = b.u32(at);
        let ends = at + 4;
        StrTable {
            b: b.clone(),
            count,
            ends,
            data: ends + 4 * (count as usize + 1),
        }
    }

    pub fn len(&self) -> u32 {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    #[inline]
    pub fn get(&self, i: u32) -> &str {
        let s = self.b.u32(self.ends + 4 * i as usize) as usize;
        let e = self.b.u32(self.ends + 4 * (i as usize + 1)) as usize;
        // SAFETY: table bytes were written from `&str` values, and the whole
        // file passed its CRC32 check before it was cached and mapped.
        unsafe { std::str::from_utf8_unchecked(&self.b.0[self.data + s..self.data + e]) }
    }

    /// Index of `key` in a sorted table.
    pub fn find(&self, key: &str) -> Option<u32> {
        let (mut lo, mut hi) = (0u32, self.count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.get(mid).cmp(key) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return Some(mid),
            }
        }
        None
    }

    /// First index at or after the bound in a sorted table.
    pub fn lower_bound(&self, from: Bound<&str>) -> u32 {
        let (key, inclusive) = match from {
            Bound::Unbounded => return 0,
            Bound::Included(k) => (k, true),
            Bound::Excluded(k) => (k, false),
        };
        let (mut lo, mut hi) = (0u32, self.count);
        while lo < hi {
            let mid = (lo + hi) / 2;
            let before = match self.get(mid).cmp(key) {
                Ordering::Less => true,
                Ordering::Equal => !inclusive,
                Ordering::Greater => false,
            };
            if before { lo = mid + 1 } else { hi = mid }
        }
        lo
    }
}

/// Decodes one term's text postings: doc ids with their positions.
pub struct TextPostings<'a> {
    buf: &'a [u8],
    pos: usize,
    doc: u32,
}

impl TextPostings<'_> {
    /// Advances to the next document, filling `positions`.
    #[inline]
    pub fn next(&mut self, positions: &mut Vec<u32>) -> Option<u32> {
        if self.pos >= self.buf.len() {
            return None;
        }
        self.doc += get_varint(self.buf, &mut self.pos);
        let freq = get_varint(self.buf, &mut self.pos);
        positions.clear();
        let mut p = 0;
        for _ in 0..freq {
            p += get_varint(self.buf, &mut self.pos);
            positions.push(p);
        }
        Some(self.doc)
    }
}

pub struct TextField {
    b: Bytes,
    pub docs_with_field: u32,
    pub total_tokens: u64,
    lengths: usize,
    pub terms: StrTable,
    df: usize,
    ends: usize,
    postings: usize,
}

impl TextField {
    fn open(b: &Bytes, at: usize) -> TextField {
        let slot = |i: usize| b.usize(at + 12 + i * 8);
        TextField {
            b: b.clone(),
            docs_with_field: b.u32(at),
            total_tokens: b.u64(at + 4),
            lengths: slot(0),
            terms: StrTable::open(b, slot(1)),
            df: slot(2),
            ends: slot(3),
            postings: slot(4),
        }
    }

    #[inline]
    pub fn length(&self, doc: u32) -> u32 {
        self.b.u32(self.lengths + 4 * doc as usize)
    }

    pub fn df(&self, ord: u32) -> u32 {
        self.b.u32(self.df + 4 * ord as usize)
    }

    pub fn postings(&self, ord: u32) -> TextPostings<'_> {
        let s = self.b.u32(self.ends + 4 * ord as usize) as usize;
        let e = self.b.u32(self.ends + 4 * (ord as usize + 1)) as usize;
        TextPostings {
            buf: &self.b.0[self.postings + s..self.postings + e],
            pos: 0,
            doc: 0,
        }
    }
}

pub struct StrField {
    b: Bytes,
    pub terms: StrTable,
    df: usize,
    ends: usize,
    postings: usize,
    first: usize,
}

impl StrField {
    fn open(b: &Bytes, at: usize) -> StrField {
        let slot = |i: usize| b.usize(at + i * 8);
        StrField {
            b: b.clone(),
            terms: StrTable::open(b, slot(0)),
            df: slot(1),
            ends: slot(2),
            postings: slot(3),
            first: slot(4),
        }
    }

    pub fn df(&self, ord: u32) -> u32 {
        self.b.u32(self.df + 4 * ord as usize)
    }

    pub fn docs(&self, ord: u32) -> impl Iterator<Item = u32> + '_ {
        let s = self.b.u32(self.ends + 4 * ord as usize) as usize;
        let e = self.b.u32(self.ends + 4 * (ord as usize + 1)) as usize;
        let buf = &self.b.0[self.postings + s..self.postings + e];
        let (mut pos, mut doc) = (0usize, 0u32);
        std::iter::from_fn(move || {
            if pos >= buf.len() {
                return None;
            }
            doc += get_varint(buf, &mut pos);
            Some(doc)
        })
    }

    /// The document's first value, used for sorting and existence checks.
    #[inline]
    pub fn first(&self, doc: u32) -> Option<&str> {
        self.first_ord(doc).map(|ord| self.terms.get(ord))
    }

    /// The ordinal of the document's first value. Terms are stored sorted, so
    /// within one segment ordinals compare exactly as the strings do.
    #[inline]
    pub fn first_ord(&self, doc: u32) -> Option<u32> {
        match self.b.u32(self.first + 4 * doc as usize) {
            NO_ORD => None,
            ord => Some(ord),
        }
    }
}

pub struct NumField {
    b: Bytes,
    float: bool,
    offs: usize,
    vals: usize,
}

impl NumField {
    #[inline]
    pub fn values(&self, doc: u32) -> impl Iterator<Item = Num> + '_ {
        let s = self.b.u32(self.offs + 4 * doc as usize) as usize;
        let e = self.b.u32(self.offs + 4 * (doc as usize + 1)) as usize;
        (s..e).map(move |i| {
            let raw = self.b.u64(self.vals + 8 * i);
            if self.float {
                Num::F(f64::from_bits(raw))
            } else {
                Num::I(raw as i64)
            }
        })
    }
}

enum FieldReader {
    Text(TextField),
    Str(StrField),
    Num(NumField),
}

/// Filter results kept per segment. Bounded; evicts the oldest entry.
#[derive(Default)]
struct FilterCache {
    map: HashMap<String, Arc<Vec<u64>>>,
    order: std::collections::VecDeque<String>,
}

const FILTER_CACHE_ENTRIES: usize = 256;

pub struct Segment {
    b: Bytes,
    doc_count: u32,
    pub ids: StrTable,
    stored: usize,
    fields: HashMap<String, FieldReader>,
    filters: std::sync::Mutex<FilterCache>,
}

impl Segment {
    /// Opens verified segment bytes. Call [`verify`] first on untrusted input.
    pub fn open(backing: Backing) -> Result<Segment, String> {
        if backing.len() < 24 || &backing[..4] != MAGIC {
            return Err("not a segment file".into());
        }
        let b = Bytes(Arc::new(backing));
        let doc_count = b.u32(8);
        let dir = b.usize(12);
        let ids = StrTable::open(&b, b.usize(dir));
        let stored = b.usize(dir + 8);
        let n_fields = b.u32(dir + 16);
        let mut at = dir + 20;
        let mut fields = HashMap::new();
        for _ in 0..n_fields {
            let len = u16::from_le_bytes(b.0[at..at + 2].try_into().unwrap()) as usize;
            let name = std::str::from_utf8(&b.0[at + 2..at + 2 + len])
                .map_err(|e| e.to_string())?
                .to_string();
            let kind = b.0[at + 2 + len];
            let off = b.usize(at + 3 + len);
            at += 11 + len;
            let reader = match kind {
                0 => FieldReader::Text(TextField::open(&b, off)),
                1 => FieldReader::Str(StrField::open(&b, off)),
                2 | 3 => FieldReader::Num(NumField {
                    b: b.clone(),
                    float: kind == 3,
                    offs: b.usize(off),
                    vals: b.usize(off + 8),
                }),
                k => return Err(format!("unknown field kind {k}")),
            };
            fields.insert(name, reader);
        }
        Ok(Segment {
            b,
            doc_count,
            ids,
            stored,
            fields,
            filters: Default::default(),
        })
    }

    pub fn doc_count(&self) -> u32 {
        self.doc_count
    }

    pub fn text(&self, field: &str) -> Option<&TextField> {
        match self.fields.get(field)? {
            FieldReader::Text(t) => Some(t),
            _ => None,
        }
    }

    pub fn text_fields(&self) -> impl Iterator<Item = (&str, &TextField)> {
        self.fields.iter().filter_map(|(name, f)| match f {
            FieldReader::Text(t) => Some((name.as_str(), t)),
            _ => None,
        })
    }

    pub fn strings(&self, field: &str) -> Option<&StrField> {
        match self.fields.get(field)? {
            FieldReader::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn numbers(&self, field: &str) -> Option<&NumField> {
        match self.fields.get(field)? {
            FieldReader::Num(n) => Some(n),
            _ => None,
        }
    }

    /// The cached bitset for a filter, computing it on first use. Segments are
    /// immutable, so an entry never goes stale; deletions are applied separately.
    pub fn cached_filter(&self, key: &str, compute: impl FnOnce() -> Vec<u64>) -> Arc<Vec<u64>> {
        if let Some(bits) = self.filters.lock().unwrap().map.get(key) {
            return bits.clone();
        }
        let bits = Arc::new(compute());
        let mut cache = self.filters.lock().unwrap();
        if !cache.map.contains_key(key) {
            if cache.order.len() >= FILTER_CACHE_ENTRIES
                && let Some(old) = cache.order.pop_front()
            {
                cache.map.remove(&old);
            }
            cache.order.push_back(key.to_string());
            cache.map.insert(key.to_string(), bits.clone());
        }
        bits
    }

    /// The stored documents for `docs`, in the same order, decompressing each
    /// block only once.
    pub fn stored_many(&self, docs: &[u32]) -> Vec<Map<String, Value>> {
        let block_size = self.b.u32(self.stored) as usize;
        let mut out = vec![Map::new(); docs.len()];
        let mut order: Vec<usize> = (0..docs.len()).collect();
        order.sort_by_key(|&i| docs[i]);
        let mut current: Option<(usize, Vec<u8>)> = None;
        for i in order {
            let block = docs[i] as usize / block_size;
            if current.as_ref().is_none_or(|(b, _)| *b != block) {
                current = Some((block, self.block(block)));
            }
            let raw = &current.as_ref().unwrap().1;
            out[i] = parse_stored(raw, docs[i] as usize % block_size);
        }
        out
    }

    /// Decompresses the block holding `doc` and returns that document.
    pub fn stored(&self, doc: u32) -> Map<String, Value> {
        self.stored_many(&[doc]).pop().unwrap_or_default()
    }

    fn block(&self, block: usize) -> Vec<u8> {
        let ends = self.stored + 8;
        let n_blocks = self.b.u32(self.stored + 4) as usize;
        let data = ends + 8 * (n_blocks + 1);
        let (s, e) = (
            self.b.usize(ends + 8 * block),
            self.b.usize(ends + 8 * (block + 1)),
        );
        zstd::stream::decode_all(&self.b.0[data + s..data + e]).unwrap_or_default()
    }
}

/// The `index`-th document in a decompressed stored block.
fn parse_stored(raw: &[u8], index: usize) -> Map<String, Value> {
    let mut at = 0;
    for _ in 0..index {
        at += 4 + u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
    }
    let len = u32::from_le_bytes(raw[at..at + 4].try_into().unwrap()) as usize;
    match serde_json::from_slice(&raw[at + 4..at + 4 + len]) {
        Ok(Value::Object(m)) => m,
        _ => Map::new(),
    }
}
