extern crate pyo3;
use std::collections::HashMap;
use pyo3::prelude::*;
use bm25::BM25;
use text::sliding_text_sequence;

#[pyfunction]
fn sliding_texts(_py: Python, texts:Vec<String>, window_size:usize, stride:usize) -> PyResult<Vec<Vec<String>>>{
    Ok(sliding_text_sequence(texts, window_size, stride))
}


#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sliding_texts))?;
    m.add_class::<BM25>()?;
    Ok(())
}
