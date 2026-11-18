//! Lucene ("standard", `defType=lucene`) query syntax parser.
//!
//! Supported: `field:term`, bare terms against the default field, `"phrases"~slop`,
//! `AND`/`OR`/`NOT`/`&&`/`||`/`!`, `+`/`-`, `( )` groups including `field:( )`,
//! `[a TO b]` / `{a TO b}` ranges with `*` bounds, `*` and `?` wildcards,
//! `^boost`, `^=constant`, `*:*`, and backslash escapes.
//! Not yet: fuzzy (`term~2`), regex (`/re/`), local params (`{!...}`).

use crate::analysis::tokenize;
use crate::schema::{FieldType, Schema};
use std::ops::Bound;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Occur {
    Must,
    Should,
    MustNot,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Query {
    MatchAll,
    MatchNone,
    Term {
        field: String,
        term: String,
    },
    Phrase {
        field: String,
        terms: Vec<String>,
        slop: u32,
    },
    Prefix {
        field: String,
        prefix: String,
    },
    Wildcard {
        field: String,
        pattern: String,
    },
    Range {
        field: String,
        lower: Bound<String>,
        upper: Bound<String>,
    },
    Bool(Vec<(Occur, Query)>),
    Boost(Box<Query>, f32),
    Const(Box<Query>, f32),
}

#[derive(Debug, Clone)]
pub struct ParseOptions {
    /// Solr `df`.
    pub default_field: String,
    /// Solr `q.op=AND`.
    pub default_and: bool,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct ParseError(pub String);

fn err<T>(msg: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError(msg.into()))
}

pub fn parse(input: &str, schema: &Schema, opts: &ParseOptions) -> Result<Query, ParseError> {
    let mut p = Parser {
        chars: input.chars().collect(),
        pos: 0,
        schema,
        opts,
    };
    let q = p.parse_clauses(&opts.default_field, false)?;
    p.skip_ws();
    if p.pos < p.chars.len() {
        return err(format!(
            "unexpected '{}' at position {}",
            p.chars[p.pos], p.pos
        ));
    }
    Ok(q)
}

#[derive(PartialEq)]
enum Conj {
    None,
    And,
    Or,
}

#[derive(PartialEq)]
enum Mod {
    None,
    Req,
    Not,
}

struct Parser<'a> {
    chars: Vec<char>,
    pos: usize,
    schema: &'a Schema,
    opts: &'a ParseOptions,
}

const SPECIAL: &[char] = &['(', ')', '[', ']', '{', '}', '"', ':', '^', '~'];

impl Parser<'_> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.pos += 1;
        }
    }

    /// Consumes `word` if it appears next as a standalone keyword.
    fn eat_keyword(&mut self, word: &str) -> bool {
        let w: Vec<char> = word.chars().collect();
        let end = self.pos + w.len();
        if end > self.chars.len() || self.chars[self.pos..end] != w[..] {
            return false;
        }
        let next = self.chars.get(end).copied();
        let symbolic = !word.chars().next().unwrap().is_alphabetic();
        if symbolic || next.is_none_or(|c| c.is_whitespace() || c == '(') {
            self.pos = end;
            return true;
        }
        false
    }

    fn parse_clauses(&mut self, field: &str, in_group: bool) -> Result<Query, ParseError> {
        let mut clauses: Vec<(Occur, Query)> = vec![];
        loop {
            self.skip_ws();
            match self.peek() {
                None if in_group => return err("missing closing ')'"),
                None => break,
                Some(')') if in_group => break,
                Some(')') => return err(format!("unexpected ')' at position {}", self.pos)),
                _ => {}
            }
            let conj = if self.eat_keyword("AND") || self.eat_keyword("&&") {
                Conj::And
            } else if self.eat_keyword("OR") || self.eat_keyword("||") {
                Conj::Or
            } else {
                Conj::None
            };
            self.skip_ws();
            let modifier = if self.peek() == Some('+') {
                self.pos += 1;
                Mod::Req
            } else if self.peek() == Some('-') || self.peek() == Some('!') {
                self.pos += 1;
                Mod::Not
            } else if self.eat_keyword("NOT") {
                Mod::Not
            } else {
                Mod::None
            };
            self.skip_ws();
            let q = self.parse_clause(field)?;
            self.add_clause(&mut clauses, conj, modifier, q);
        }
        Ok(match clauses.len() {
            0 => Query::MatchNone,
            1 if clauses[0].0 != Occur::MustNot => clauses.pop().unwrap().1,
            _ => Query::Bool(clauses),
        })
    }

    /// Lucene classic QueryParser's `addClause` occurrence rules.
    fn add_clause(
        &self,
        clauses: &mut Vec<(Occur, Query)>,
        conj: Conj,
        modifier: Mod,
        q: Option<Query>,
    ) {
        if let Some(last) = clauses.last_mut() {
            if conj == Conj::And && last.0 != Occur::MustNot {
                last.0 = Occur::Must;
            }
            if self.opts.default_and && conj == Conj::Or && last.0 != Occur::MustNot {
                last.0 = Occur::Should;
            }
        }
        let Some(q) = q else { return };
        let prohibited = modifier == Mod::Not;
        let required = if self.opts.default_and {
            !prohibited && conj != Conj::Or
        } else {
            modifier == Mod::Req || (conj == Conj::And && !prohibited)
        };
        let occur = if prohibited {
            Occur::MustNot
        } else if required {
            Occur::Must
        } else {
            Occur::Should
        };
        clauses.push((occur, q));
    }

    /// Reads an unquoted token, resolving escapes. Returns (text, had_unescaped_wildcard).
    fn read_token(&mut self) -> (String, String, bool) {
        let mut text = String::new();
        let mut raw = String::new();
        let mut wild = false;
        while let Some(c) = self.peek() {
            if c == '\\'
                && let Some(&n) = self.chars.get(self.pos + 1)
            {
                text.push(n);
                raw.push('\\');
                raw.push(n);
                self.pos += 2;
                continue;
            }
            if c.is_whitespace() || SPECIAL.contains(&c) {
                break;
            }
            if c == '*' || c == '?' {
                wild = true;
            }
            text.push(c);
            raw.push(c);
            self.pos += 1;
        }
        (text, raw, wild)
    }

    fn parse_clause(&mut self, default_field: &str) -> Result<Option<Query>, ParseError> {
        let mut field = default_field.to_string();
        let start = self.pos;
        let q = if matches!(self.peek(), Some('(' | '"' | '[' | '{')) {
            self.parse_value(&field)?
        } else {
            let (text, raw, wild) = self.read_token();
            if self.peek() == Some(':') && !text.is_empty() {
                self.pos += 1;
                if text == "*" {
                    if self.peek() == Some('*') {
                        self.pos += 1;
                        return self.postfix(Some(Query::MatchAll));
                    }
                    return err("expected '*' after '*:'");
                }
                field = text;
                self.require_field(&field)?;
                self.parse_value(&field)?
            } else if text.is_empty() {
                return err(format!(
                    "unexpected '{}' at position {}",
                    self.peek().unwrap_or(' '),
                    start
                ));
            } else {
                self.require_field(&field)?;
                self.term_query(&field, &text, &raw, wild)?
            }
        };
        self.postfix(q)
    }

    fn require_field(&self, field: &str) -> Result<FieldType, ParseError> {
        match self.schema.field(field) {
            Some(info) => Ok(info.ty),
            None => err(format!("undefined field {field}")),
        }
    }

    fn parse_value(&mut self, field: &str) -> Result<Option<Query>, ParseError> {
        match self.peek() {
            Some('(') => {
                self.pos += 1;
                let q = self.parse_clauses(field, true)?;
                self.pos += 1; // ')'
                Ok(Some(q))
            }
            Some('"') => {
                self.pos += 1;
                let mut text = String::new();
                loop {
                    match self.peek() {
                        None => return err("missing closing '\"'"),
                        Some('"') => {
                            self.pos += 1;
                            break;
                        }
                        Some('\\') if self.pos + 1 < self.chars.len() => {
                            text.push(self.chars[self.pos + 1]);
                            self.pos += 2;
                        }
                        Some(c) => {
                            text.push(c);
                            self.pos += 1;
                        }
                    }
                }
                let mut slop = 0;
                if self.peek() == Some('~') {
                    self.pos += 1;
                    slop = self.read_number()?.max(0.0) as u32;
                }
                self.phrase_query(field, &text, slop)
            }
            Some(open @ ('[' | '{')) => {
                self.pos += 1;
                self.skip_ws();
                let lo = self.read_range_endpoint()?;
                self.skip_ws();
                if !self.eat_keyword("TO") {
                    return err("expected TO in range");
                }
                self.skip_ws();
                let hi = self.read_range_endpoint()?;
                self.skip_ws();
                let close = match self.peek() {
                    Some(c @ (']' | '}')) => c,
                    _ => return err("missing closing ']' or '}' in range"),
                };
                self.pos += 1;
                let ty = self.require_field(field)?;
                let norm = |s: String| {
                    if ty == FieldType::Text {
                        s.to_lowercase()
                    } else {
                        s
                    }
                };
                let bound = |s: String, inclusive: bool| match s.as_str() {
                    "*" => Bound::Unbounded,
                    _ if inclusive => Bound::Included(norm(s)),
                    _ => Bound::Excluded(norm(s)),
                };
                Ok(Some(Query::Range {
                    field: field.to_string(),
                    lower: bound(lo, open == '['),
                    upper: bound(hi, close == ']'),
                }))
            }
            _ => {
                let (text, raw, wild) = self.read_token();
                if text.is_empty() {
                    return err(format!("expected a value for field {field}"));
                }
                self.term_query(field, &text, &raw, wild)
            }
        }
    }

    /// Range endpoints may contain `:` and other specials (dates), as in
    /// Lucene's RANGE_GOOP; they end at whitespace or a closing bracket.
    /// A quoted endpoint is taken literally.
    fn read_range_endpoint(&mut self) -> Result<String, ParseError> {
        let mut out = String::new();
        if self.peek() == Some('"') {
            self.pos += 1;
            loop {
                match self.peek() {
                    None => return err("missing closing '\"' in range"),
                    Some('"') => {
                        self.pos += 1;
                        return Ok(out);
                    }
                    Some('\\') if self.pos + 1 < self.chars.len() => {
                        out.push(self.chars[self.pos + 1]);
                        self.pos += 2;
                    }
                    Some(c) => {
                        out.push(c);
                        self.pos += 1;
                    }
                }
            }
        }
        while let Some(c) = self.peek() {
            if c.is_whitespace() || c == ']' || c == '}' {
                break;
            }
            if c == '\\' && self.pos + 1 < self.chars.len() {
                out.push(self.chars[self.pos + 1]);
                self.pos += 2;
                continue;
            }
            out.push(c);
            self.pos += 1;
        }
        if out.is_empty() {
            return err("empty range endpoint");
        }
        Ok(out)
    }

    fn read_number(&mut self) -> Result<f32, ParseError> {
        let start = self.pos;
        while self.peek().is_some_and(|c| c.is_ascii_digit() || c == '.') {
            self.pos += 1;
        }
        let s: String = self.chars[start..self.pos].iter().collect();
        s.parse()
            .map_err(|_| ParseError(format!("expected a number at position {start}")))
    }

    fn postfix(&mut self, q: Option<Query>) -> Result<Option<Query>, ParseError> {
        let mut q = q;
        loop {
            match self.peek() {
                Some('^') if self.chars.get(self.pos + 1) == Some(&'=') => {
                    self.pos += 2;
                    let n = self.read_number()?;
                    q = q.map(|q| Query::Const(Box::new(q), n));
                }
                Some('^') => {
                    self.pos += 1;
                    let n = self.read_number()?;
                    q = q.map(|q| Query::Boost(Box::new(q), n));
                }
                Some('~') => return err("fuzzy queries (term~N) are not supported yet"),
                _ => return Ok(q),
            }
        }
    }

    fn analysed_tokens_query(&self, field: &str, tokens: Vec<String>) -> Option<Query> {
        let occur = if self.opts.default_and {
            Occur::Must
        } else {
            Occur::Should
        };
        let mut terms: Vec<Query> = tokens
            .into_iter()
            .map(|t| Query::Term {
                field: field.into(),
                term: t,
            })
            .collect();
        match terms.len() {
            0 => None,
            1 => terms.pop(),
            _ => Some(Query::Bool(terms.into_iter().map(|t| (occur, t)).collect())),
        }
    }

    fn term_query(
        &self,
        field: &str,
        text: &str,
        raw: &str,
        wild: bool,
    ) -> Result<Option<Query>, ParseError> {
        let ty = self.require_field(field)?;
        if wild {
            let pattern = if ty == FieldType::Text {
                text.to_lowercase()
            } else {
                text.to_string()
            };
            // Escaped wildcards in `raw` are literal; only treat a single trailing
            // unescaped `*` as a prefix query.
            let body = &pattern[..pattern.len().saturating_sub(1)];
            let trailing_star = raw.ends_with('*') && !raw.ends_with("\\*");
            if trailing_star && !body.contains(['*', '?']) && !body.is_empty() {
                return Ok(Some(Query::Prefix {
                    field: field.into(),
                    prefix: body.to_string(),
                }));
            }
            return Ok(Some(Query::Wildcard {
                field: field.into(),
                pattern,
            }));
        }
        self.typed_value(field, ty, text)
    }

    fn typed_value(
        &self,
        field: &str,
        ty: FieldType,
        text: &str,
    ) -> Result<Option<Query>, ParseError> {
        match ty {
            FieldType::Text => Ok(self.analysed_tokens_query(field, tokenize(text))),
            FieldType::Str => Ok(Some(Query::Term {
                field: field.into(),
                term: text.into(),
            })),
            FieldType::Long | FieldType::Double | FieldType::Date => {
                if crate::segment::parse_numeric(ty, text).is_none() {
                    return err(format!("invalid value '{text}' for field {field}"));
                }
                Ok(Some(Query::Term {
                    field: field.into(),
                    term: text.into(),
                }))
            }
            FieldType::Ignored => Ok(Some(Query::MatchNone)),
        }
    }

    fn phrase_query(
        &self,
        field: &str,
        text: &str,
        slop: u32,
    ) -> Result<Option<Query>, ParseError> {
        let ty = self.require_field(field)?;
        if ty != FieldType::Text {
            return self.typed_value(field, ty, text);
        }
        let mut terms = tokenize(text);
        Ok(match terms.len() {
            0 => None,
            1 => Some(Query::Term {
                field: field.into(),
                term: terms.pop().unwrap(),
            }),
            _ => Some(Query::Phrase {
                field: field.into(),
                terms,
                slop,
            }),
        })
    }
}
