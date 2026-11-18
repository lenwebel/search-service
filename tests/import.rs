use search_service::import::clean_doc;
use search_service::schema::Schema;
use serde_json::json;

#[test]
fn clean_doc_drops_solr_internal_and_copy_destination_fields() {
    let schema = Schema::from_solr_dir("schemas/collectorspoke-cards").unwrap();
    let doc = json!({
        "id": "d5a3", "name": "Visiting Christmas Past", "set": "Winterspell",
        "name_str": "Visiting Christmas Past", "set_str": "Winterspell",
        "search_all": ["Winterspell"], "_version_": 1880882643692683268u64
    });
    assert_eq!(
        clean_doc(&schema, doc.as_object().unwrap().clone()),
        json!({"id": "d5a3", "name": "Visiting Christmas Past", "set": "Winterspell"})
            .as_object()
            .unwrap()
            .clone()
    );
}
