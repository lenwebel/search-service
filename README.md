# search-service

A Lucene-style full-text search engine in Rust. It stores its index on object
storage (local disk or Azure Blob) with **no single-writer lock**: any number of
instances can write concurrently, and a crashed instance can't leave a stale
lock behind. It speaks a Solr-compatible subset of `/select` and `/update` and
the Lucene query syntax.

This is a proof of concept. See [docs/design.md](docs/design.md) for the
storage and commit protocol, the segment format, what's supported, and what
isn't yet.

## Run

```sh
cargo test
STORE_URL=file:///tmp/search-index cargo run --release -- serve
curl -X POST localhost:8983/solr/collectorspoke-cards/update \
  -H 'content-type: application/json' \
  -d '[{"id":"1","name":"Pikachu ex"}]'
curl 'localhost:8983/solr/collectorspoke-cards/select?q=name:pikachu'
```

To copy an existing Solr core:

```sh
SOLR_AUTH_CREDENTIALS=<base64 user:pass> STORE_URL=file:///tmp/search-index \
  cargo run --release -- import --from https://<host>/solr/<core>
```

The schema is read from a Solr Schema API export in `schemas/<core>/`.
Benchmarks are in `examples/bench.rs`.
