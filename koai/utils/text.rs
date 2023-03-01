extern crate pyo3;
use pyo3::prelude::*;

fn sliding_text_sequence(tokens:Vec<String>, window_size:usize, stride:usize) -> Vec<Vec<String>>{
    let length = ((tokens.len() - window_size) as f32 /stride as f32).ceil() as usize + 1;
    let token_length = tokens.len();

    let mut outputs = Vec::new();
    for i in 0..length {
        let start = i+(stride-1)*i;
        let end = start + window_size;
        outputs.push(tokens[start..end.min(token_length)].to_vec());
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