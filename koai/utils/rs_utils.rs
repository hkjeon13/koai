extern crate pyo3;
use std::collections::HashMap;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

trait Entry {
    fn add_neighbour(&mut self, neighbour: String);

}

trait Searcher {
    fn search(&self, tokenized_query: Vec<String>, n: i32) -> Vec<(String, f32)>;
    fn add_document(&mut self, id:String, doc: String, tokenized_doc: Vec<String>);
    fn remove_document(&mut self, id:String);
    fn _calculate(&self, tokenized_query: Vec<String>, doc: &Document, avg_doc_length:f32) -> f32;
}


struct Token {
    text: String,
    maps: HashMap<String, i32>,
}

struct Document {
    id: String,
    text: String,
    maps: HashMap<String, i32>,
}


impl Entry for Token {
    fn add_neighbour(&mut self, neighbour: String) {
        if self.maps.contains_key(&neighbour) {
            *self.maps.get_mut(&neighbour).unwrap() += 1;
        } else {
            self.maps.insert(neighbour, 1);
        }
    }
}

impl Clone for Token {
    fn clone(&self) -> Self {
        Token {
            text: self.text.clone(),
            maps: self.maps.clone(),
        }
    }
}


impl Entry for Document {
    fn add_neighbour(&mut self, neighbour: String) {
        if self.maps.contains_key(&neighbour) {
            *self.maps.get_mut(&neighbour).unwrap() += 1;
        } else {
            self.maps.insert(neighbour, 1);
        }
    }
}

#[pyclass]
pub struct BM25 {
    index: HashMap<String, Document>,
    token_index: HashMap<String, Token>,
    k1: f32,
    b: f32,
}

#[pymethods]
impl Searcher for BM25 {
    #[classmethod]
    fn _calculate(&self, tokenized_query: Vec<String>, doc: &Document, avg_doc_length:f32) -> f32 {
        let N = self.index.len() as f32;

        let mut score = 0.0;
        for token in tokenized_query {
            if doc.maps.contains_key(&token) {
                let tf = *doc.maps.get(&token).unwrap() as f32;
                let mut idf = self.token_index.get(&token).unwrap().maps.len() as f32;
                idf = (((N - idf + 0.5) / (idf + 0.5))+1.0).ln();
                score += (tf * (1.0 + self.k1) / (tf + self.k1 * ((1.-self.b) + self.b * (doc.text.len() as f32 / avg_doc_length)))) * idf;
            }
        }
        score
    }
    #[classmethod]
    fn search(&self, tokenized_query: Vec<String>, n: i32) -> Vec<(String, f32)> {
        let avg_doc_length = self.index.iter().map(|(_, doc)| doc.text.len()).sum::<usize>() as f32 / self.index.len() as f32;
        let mut result = self.index.iter().map(|(id, doc)| {
            (id.to_string(), self._calculate(tokenized_query.clone(), doc, avg_doc_length))
        }).collect::<Vec<_>>();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        result
    }
    #[classmethod]
    fn add_document(&mut self, id:String, doc: String, tokenized_doc: Vec<String>) {
        if !self.index.contains_key(&id) {
            let mut document = Document{
                id: id.to_string(),
                text: doc,
                maps: HashMap::new(),
            };
            for token in tokenized_doc {
                document.add_neighbour(token.to_string());
                if !self.token_index.contains_key(token.as_str()){
                    let mut token_object = Token{
                        text: token.to_string(),
                        maps: HashMap::new(),
                    };
                    token_object.add_neighbour(id.to_string());
                    self.token_index.insert(token, token_object.clone());

                } else {
                    let token_object = self.token_index.get_mut(token.as_str()).unwrap();
                    token_object.add_neighbour(id.to_string());

                }
            }
            self.index.insert(id, document);
        }

    }
    #[classmethod]
    fn remove_document(&mut self, id:String) {
        self.index.remove(&id);
    }
}

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
    m.add_class::<BM25>()?;
    Ok(())
}
