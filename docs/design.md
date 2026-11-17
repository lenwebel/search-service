# Search service: proof of concept

A Lucene-style search engine in Rust, built to replace a Solr core whose index
lives on Azure Files and corrupts when instances change. It keeps Solr's query
language and HTTP shapes, but stores the index on object storage with no
single-writer lock.

## Storage and commits

Everything lives under `<store>/<core>/`:

| Path | Written by | Contents |
|------|-----------|----------|
| `segments/<ulid>.seg` | one writer, once | an immutable segment (postings with positions, string and numeric columns, stored JSON) |
| `deletes/<ulid>.json` | one writer, once | ids a commit deleted or replaced |
| `manifest/<generation>.json` | whoever wins that generation | the full list of live segments and delete files |

Commit protocol:

1. Build the segment in memory and upload it, and the delete file, under fresh ULID names. Writers never share files, so this needs no coordination.
2. Read the newest manifest `N`, append the new files, and create manifest `N+1` with **put-if-absent** (`PutMode::Create`; on Azure Blob this is `If-None-Match: *`).
3. If `N+1` already exists, another writer won the race. Re-read and try `N+2`.

There's no lock to go stale. A writer that dies between steps 1 and 2 leaves
unreferenced files, which a future garbage collector will remove.

Ordering: a delete or re-add of an id committed at generation G hides copies of
that id in segments from generations before G. That gives Solr's overwrite
semantics for re-added ids, and makes concurrent deletes from different
instances resolve by commit order.

Readers take a point-in-time snapshot of one manifest. Segments are immutable,
so each instance downloads a segment once, verifies its CRC32 checksum, writes
it to a local cache directory and memory-maps it. A corrupted segment is
refused rather than served. A new snapshot reuses the previous one's segments
and only applies delete files committed since.

## Segment format (v2)

There's one binary file per segment, read through a memory map with no
up-front decoding:

- sorted term dictionaries, looked up by binary search
- postings as varint-encoded doc and position deltas
- per-document field lengths, first-value ordinals for string sorting, and
  numeric columns
- stored documents compressed with zstd in blocks of 64
- a CRC32 trailer

The exact layout is documented at the top of `src/segment.rs`.

## Supported now

- Schema loaded from a Solr Schema API export (`schemas/<core>/`): fields,
  dynamic fields and copy fields. Types: text (UAX#29 tokeniser plus
  lowercase, matching `text_general`), string, long, double and date.
- `defType=lucene` syntax: terms, `field:`, phrases with slop, AND/OR/NOT/+/-,
  groups, ranges, `*` and `?` wildcards, `^boost`, `^=constant` and `*:*`.
- `GET /solr/<core>/select`: `q`, repeated `fq`, `df`, `q.op`, `sort`, `start`,
  `rows` and `fl` (including `score`), with Solr-shaped responses and errors.
- `POST /solr/<core>/update`: a JSON array of documents, a single document,
  or `add`/`delete` commands. Each request is committed when it returns.
- BM25 scoring with statistics aggregated across segments.
- Read path: segments are searched in parallel, each keeping only its top
  `start + rows` hits. `fq` results are cached per segment as bitsets (256 per
  segment). Because segments never change, cached filters survive new commits,
  unlike Solr's filterCache, which a commit discards. `fl` limited to the
  unique key skips decompressing stored documents.
- `import`: copies a live Solr core using cursorMark paging.
- Update bodies up to 512 MB, for bulk indexing.

## Not yet

In rough priority order for the current app:

- the JSON Request API (`/query`) and JSON Facet API (terms facets, nested
  facets, `excludeTags`)
- `{!tag}` local params
- segment merging and garbage collection
- dismax and edismax
- fuzzy and regex queries
- delete-by-query
- auth
- Azure Blob tested end to end (the code path exists via `STORE_URL=az://...`, but is untested)

Scores are BM25 but not bit-identical to Solr, which uses lossy 1-byte length
norms and corpus-wide statistics. Ties between equal scores are broken by
index order, which differs from Solr's.
