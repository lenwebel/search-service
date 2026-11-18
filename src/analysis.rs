//! Text analysis for `Text` fields.
//!
//! Approximates Solr's `text_general` chain: Lucene's StandardTokenizer
//! implements UAX#29 word boundaries, which `unicode-segmentation` also
//! implements, followed by lowercasing. The default `stopwords.txt` is empty,
//! so no stop filter is applied.

use unicode_segmentation::UnicodeSegmentation;

pub fn tokenize(text: &str) -> Vec<String> {
    text.unicode_words().map(|w| w.to_lowercase()).collect()
}
