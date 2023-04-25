extern crate pyo3;
use pyo3::prelude::*;


fn sliding_text_sequence(tokens:Vec<String>, window_size:usize, stride:usize) -> Vec<Vec<String>> {
    let n_iter = ((tokens.len() - window_size) as f32 / stride as f32).ceil() as usize + 1;
    let mut outputs: Vec<Vec<String>> = Vec::with_capacity(n_iter);
    for i in 0..n_iter {
        let output = tokens[i * stride..(i * stride + window_size).min(tokens.len())].to_vec();
        outputs.push(output);
    }
    outputs
}


#[pyfunction]
fn sliding_texts(_py: Python, texts:Vec<String>, window_size:usize, stride:usize) -> PyResult<Vec<Vec<String>>>{
    Ok(sliding_text_sequence(texts, window_size, stride))
}


#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(sliding_texts))?;
    Ok(())
}