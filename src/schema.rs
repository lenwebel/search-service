//! Field definitions, loaded from a Solr Schema API export so an existing core
//! can be indexed without hand-translating its schema.

use serde_json::Value;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FieldType {
    /// Analysed full text (tokenised, lowercased, positions kept).
    Text,
    /// Exact, case-sensitive string.
    Str,
    /// 64-bit integer.
    Long,
    /// 64-bit float.
    Double,
    /// Instant, stored as epoch milliseconds.
    Date,
    /// Known to the schema but not indexed by this engine (spatial, payloads, ...).
    Ignored,
}

impl FieldType {
    pub fn is_numeric(self) -> bool {
        matches!(self, FieldType::Long | FieldType::Double | FieldType::Date)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FieldInfo {
    pub ty: FieldType,
    pub multi: bool,
}

#[derive(Debug, Clone)]
pub struct Schema {
    pub unique_key: String,
    fields: Vec<(String, FieldInfo)>,
    /// Name -> position in `fields`, so lookups on hot paths are O(1).
    by_name: std::collections::HashMap<String, usize>,
    /// Patterns like `*_s` or `attr_*`, longest first (Solr's precedence).
    dynamic: Vec<(String, FieldInfo)>,
    copies: Vec<(String, String)>,
}

#[derive(Debug, thiserror::Error)]
pub enum SchemaError {
    #[error("reading schema file {0}: {1}")]
    Io(String, std::io::Error),
    #[error("parsing schema file {0}: {1}")]
    Json(String, serde_json::Error),
    #[error("schema file {0} is missing {1}")]
    Missing(String, &'static str),
}

fn glob_match(pattern: &str, name: &str) -> bool {
    if let Some(suffix) = pattern.strip_prefix('*') {
        name.ends_with(suffix) && name.len() > suffix.len()
    } else if let Some(prefix) = pattern.strip_suffix('*') {
        name.starts_with(prefix) && name.len() > prefix.len()
    } else {
        pattern == name
    }
}

fn type_from_solr_class(class: &str) -> FieldType {
    let class = class.rsplit('.').next().unwrap_or(class);
    match class {
        "TextField" | "SortableTextField" => FieldType::Text,
        "StrField" | "BoolField" | "UUIDField" => FieldType::Str,
        "LongPointField" | "IntPointField" | "TrieLongField" | "TrieIntField" => FieldType::Long,
        "DoublePointField" | "FloatPointField" | "TrieDoubleField" | "TrieFloatField" => {
            FieldType::Double
        }
        "DatePointField" | "TrieDateField" => FieldType::Date,
        _ => FieldType::Ignored,
    }
}

impl Schema {
    pub fn new(unique_key: &str) -> Self {
        Schema {
            unique_key: unique_key.into(),
            fields: vec![],
            by_name: Default::default(),
            dynamic: vec![],
            copies: vec![],
        }
    }

    pub fn with_field(mut self, name: &str, ty: FieldType, multi: bool) -> Self {
        self.by_name.insert(name.into(), self.fields.len());
        self.fields.push((name.into(), FieldInfo { ty, multi }));
        self
    }

    pub fn with_dynamic(mut self, pattern: &str, ty: FieldType, multi: bool) -> Self {
        self.dynamic.push((pattern.into(), FieldInfo { ty, multi }));
        self.dynamic.sort_by_key(|d| std::cmp::Reverse(d.0.len()));
        self
    }

    pub fn with_copy(mut self, source: &str, dest: &str) -> Self {
        self.copies.push((source.into(), dest.into()));
        self
    }

    /// Loads `fields.json`, `dyn.json`, `copy.json`, `types.json` and
    /// `uniq.json`, the raw responses of Solr's `/schema/*` endpoints.
    pub fn from_solr_dir(dir: impl AsRef<Path>) -> Result<Self, SchemaError> {
        let dir = dir.as_ref();
        let load = |name: &str| -> Result<Value, SchemaError> {
            let path = dir.join(name);
            let p = path.display().to_string();
            let text = std::fs::read_to_string(&path).map_err(|e| SchemaError::Io(p.clone(), e))?;
            serde_json::from_str(&text).map_err(|e| SchemaError::Json(p, e))
        };
        let types = load("types.json")?;
        let types = types["fieldTypes"]
            .as_array()
            .ok_or(SchemaError::Missing("types.json".into(), "fieldTypes"))?;
        let resolve = |def: &Value| -> FieldInfo {
            let ty_name = def["type"].as_str().unwrap_or_default();
            let ty_def = types.iter().find(|t| t["name"] == ty_name);
            let ty = ty_def
                .and_then(|t| t["class"].as_str())
                .map(type_from_solr_class)
                .unwrap_or(FieldType::Ignored);
            let multi = def["multiValued"]
                .as_bool()
                .or_else(|| ty_def.and_then(|t| t["multiValued"].as_bool()))
                .unwrap_or(false);
            FieldInfo { ty, multi }
        };

        let unique_key = load("uniq.json")?["uniqueKey"]
            .as_str()
            .unwrap_or("id")
            .to_string();
        let mut schema = Schema::new(&unique_key);
        for f in load("fields.json")?["fields"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(name) = f["name"].as_str() {
                schema.by_name.insert(name.into(), schema.fields.len());
                schema.fields.push((name.into(), resolve(f)));
            }
        }
        for f in load("dyn.json")?["dynamicFields"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let Some(name) = f["name"].as_str() {
                schema.dynamic.push((name.into(), resolve(f)));
            }
        }
        schema.dynamic.sort_by_key(|d| std::cmp::Reverse(d.0.len()));
        for c in load("copy.json")?["copyFields"]
            .as_array()
            .into_iter()
            .flatten()
        {
            if let (Some(s), Some(d)) = (c["source"].as_str(), c["dest"].as_str()) {
                schema.copies.push((s.into(), d.into()));
            }
        }
        Ok(schema)
    }

    pub fn field(&self, name: &str) -> Option<FieldInfo> {
        if let Some(&i) = self.by_name.get(name) {
            return Some(self.fields[i].1.clone());
        }
        self.dynamic
            .iter()
            .find(|(p, _)| glob_match(p, name))
            .map(|(_, info)| info.clone())
    }

    /// Destination fields that `name` is copied into at index time.
    pub fn copy_targets(&self, name: &str) -> Vec<String> {
        self.copies
            .iter()
            .filter(|(s, _)| glob_match(s, name))
            .map(|(_, d)| d.clone())
            .collect()
    }

    /// True if `name` only exists as a copy destination, so documents read
    /// back from Solr can drop it and let this engine recompute it.
    pub fn is_copy_destination(&self, name: &str) -> bool {
        self.copies.iter().any(|(_, d)| d == name)
    }
}
