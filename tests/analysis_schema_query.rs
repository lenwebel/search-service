use search_service::analysis::tokenize;
use search_service::query::{Occur, ParseOptions, Query, parse};
use search_service::schema::{FieldType, Schema};
use std::ops::Bound;

fn test_schema() -> Schema {
    Schema::new("id")
        .with_field("id", FieldType::Str, false)
        .with_field("name", FieldType::Text, false)
        .with_field("name_str", FieldType::Str, false)
        .with_field("tags", FieldType::Str, true)
        .with_field("hitpoints", FieldType::Long, false)
        .with_field("released", FieldType::Date, false)
        .with_copy("name", "name_str")
}

fn opts() -> ParseOptions {
    ParseOptions {
        default_field: "name".into(),
        default_and: false,
    }
}

fn term(f: &str, t: &str) -> Query {
    Query::Term {
        field: f.into(),
        term: t.into(),
    }
}

#[test]
fn tokenize_splits_on_unicode_word_boundaries_and_lowercases() {
    assert_eq!(
        tokenize("Visiting Christmas Past (Cold Foil)"),
        vec!["visiting", "christmas", "past", "cold", "foil"]
    );
    assert_eq!(tokenize("162/204"), vec!["162", "204"]);
    assert_eq!(
        tokenize("Mickey's Christmas"),
        vec!["mickey's", "christmas"]
    );
}

#[test]
fn schema_resolves_explicit_dynamic_and_copy_fields_from_solr_export() {
    let s = Schema::from_solr_dir("schemas/collectorspoke-cards").unwrap();
    assert_eq!(s.unique_key, "id");
    assert_eq!(s.field("name").unwrap().ty, FieldType::Text);
    assert_eq!(s.field("name_str").unwrap().ty, FieldType::Str);
    assert_eq!(s.field("hitpoints").unwrap().ty, FieldType::Long);
    assert_eq!(s.field("anything_dt").unwrap().ty, FieldType::Date);
    assert_eq!(s.field("anything_s").unwrap().ty, FieldType::Str);
    assert!(s.field("no_such_field_xyz").is_none());
    let targets = s.copy_targets("name");
    assert!(targets.contains(&"name_str".to_string()));
    assert!(targets.contains(&"search_all".to_string()));
}

#[test]
fn parses_bare_terms_against_default_field_with_or_semantics() {
    let q = parse("Pikachu ex", &test_schema(), &opts()).unwrap();
    assert_eq!(
        q,
        Query::Bool(vec![
            (Occur::Should, term("name", "pikachu")),
            (Occur::Should, term("name", "ex"))
        ])
    );
}

#[test]
fn and_or_not_and_modifiers_follow_lucene_classic_rules() {
    let q = parse("a AND b OR -c", &test_schema(), &opts()).unwrap();
    assert_eq!(
        q,
        Query::Bool(vec![
            (Occur::Must, term("name", "a")),
            (Occur::Must, term("name", "b")),
            (Occur::MustNot, term("name", "c")),
        ])
    );
    let q = parse("+a NOT b", &test_schema(), &opts()).unwrap();
    assert_eq!(
        q,
        Query::Bool(vec![
            (Occur::Must, term("name", "a")),
            (Occur::MustNot, term("name", "b"))
        ])
    );
}

#[test]
fn string_fields_are_exact_and_quoted_wildcards_stay_literal() {
    let s = test_schema();
    assert_eq!(
        parse("name_str:\"Island (Non-Foil)\"", &s, &opts()).unwrap(),
        term("name_str", "Island (Non-Foil)")
    );
    assert_eq!(
        parse("name_str:\"Pika*\"", &s, &opts()).unwrap(),
        term("name_str", "Pika*")
    );
    assert_eq!(
        parse("name_str:Pika*", &s, &opts()).unwrap(),
        Query::Prefix {
            field: "name_str".into(),
            prefix: "Pika".into()
        }
    );
    assert_eq!(
        parse("name_str:*ika*", &s, &opts()).unwrap(),
        Query::Wildcard {
            field: "name_str".into(),
            pattern: "*ika*".into()
        }
    );
}

#[test]
fn quoted_text_becomes_phrase_with_optional_slop() {
    let s = test_schema();
    assert_eq!(
        parse("name:\"Cold Foil\"~2", &s, &opts()).unwrap(),
        Query::Phrase {
            field: "name".into(),
            terms: vec!["cold".into(), "foil".into()],
            slop: 2
        }
    );
}

#[test]
fn ranges_boosts_constant_scores_groups_and_match_all() {
    let s = test_schema();
    assert_eq!(
        parse("hitpoints:[100 TO *}", &s, &opts()).unwrap(),
        Query::Range {
            field: "hitpoints".into(),
            lower: Bound::Included("100".into()),
            upper: Bound::Unbounded
        }
    );
    assert_eq!(
        parse("name:foil^2", &s, &opts()).unwrap(),
        Query::Boost(Box::new(term("name", "foil")), 2.0)
    );
    assert_eq!(
        parse("(name_str:\"X\")^=100", &s, &opts()).unwrap(),
        Query::Const(Box::new(term("name_str", "X")), 100.0)
    );
    assert_eq!(parse("*:*", &s, &opts()).unwrap(), Query::MatchAll);
    assert_eq!(
        parse("tags:(a OR b)", &s, &opts()).unwrap(),
        Query::Bool(vec![
            (Occur::Should, term("tags", "a")),
            (Occur::Should, term("tags", "b"))
        ])
    );
}

#[test]
fn escapes_and_multi_token_terms_and_errors() {
    let s = test_schema();
    assert_eq!(
        parse("name_str:Non\\-Foil\\:1", &s, &opts()).unwrap(),
        term("name_str", "Non-Foil:1")
    );
    // analysed text term producing several tokens becomes a boolean of those tokens
    assert_eq!(
        parse("name:162/204", &s, &opts()).unwrap(),
        Query::Bool(vec![
            (Occur::Should, term("name", "162")),
            (Occur::Should, term("name", "204"))
        ])
    );
    assert!(parse("name:(unclosed", &s, &opts()).is_err());
    assert!(parse("nosuchfield:x", &s, &opts()).is_err());
}
