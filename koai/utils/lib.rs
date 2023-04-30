extern crate pyo3;
use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use serde_derive::{Serialize,Deserialize};
use std::path::Path;
use tqdm_rs;
use counter::Counter;


#[derive(Serialize, Deserialize, Debug)]
struct Token {
    text: String,
    maps: HashMap<String, i32>,
}

#[pyclass]
#[derive(Serialize, Deserialize, Debug)]
struct Document {
    id: String,
    maps: HashMap<String, i32>,
}

impl Token {
    fn add_neighbour(&mut self, neighbour: &str, value:i32) {
        if self.maps.contains_key(neighbour) {
            *self.maps.get_mut(neighbour).unwrap() += value;
        } else {
            self.maps.insert(neighbour.to_string(), value);
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

#[pymethods]
impl Document {
    fn add_neighbour(&mut self, neighbour: &str, value:i32) {
        if self.maps.contains_key(neighbour) {
            *self.maps.get_mut(neighbour).unwrap() += value;
        } else {
            self.maps.insert(neighbour.to_string(), value);
        }
    }
    fn len(&self) -> usize {
        self.maps.values().sum::<i32>() as usize
    }
}

#[pyclass]
pub struct BM25 {
    index: HashMap<String, Document>,
    token_index: HashMap<String, Token>,
    map_bm25: HashMap<String, f32>,
    k1: f32,
    b: f32,
}

fn _calculate(tf: f32, num_docs:f32, doc_len: usize, average_length: f32, k1: f32, b: f32, idf: f32) -> f32 {
    (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len as f32 / average_length))) * (((num_docs as f32 - idf + 0.5) / (idf + 0.5))+1.0).ln()
}

#[pymethods]
impl BM25 {
    #[new]
    fn new() -> Self {
        BM25 {
            index: HashMap::new(),
            token_index: HashMap::new(),
            map_bm25: HashMap::new(),
            k1: 1.2,
            b: 0.75,
        }
    }

    fn save_index(&self, path: &str) {
        let json = serde_json::to_string(&self.index).unwrap();
        std::fs::write(path, json).expect("Unable to write file");
    }

    fn save_token_index(&self, path: &str) {
        let json = serde_json::to_string(&self.token_index).unwrap();
        std::fs::write(path, json).expect("Unable to write file");
    }

    fn save(&self, save_directory: &str) {
        self.save_index(Path::new(save_directory).join("index.json").to_str().unwrap());
        self.save_token_index(Path::new(save_directory).join("token_index.json").to_str().unwrap());
        println!("Saved index with {} documents and {} tokens", self.index.len(), self.token_index.len());
    }

    fn load(&mut self, load_directory: &str) {
        let index_path = Path::new(load_directory).join("index.json");
        let token_index_path = Path::new(load_directory).join("token_index.json");
        let index_json = std::fs::read_to_string(index_path).expect("Unable to read file");
        let token_index_json = std::fs::read_to_string(token_index_path).expect("Unable to read file");
        self.index = serde_json::from_str(&index_json).unwrap();
        self.token_index = serde_json::from_str(&token_index_json).unwrap();
        println!("Loaded index with {} documents and {} tokens", self.index.len(), self.token_index.len());

    }

    fn freeze(&mut self) {
        println!("freezing index");
        let mut map_bm25 = HashMap::new();
        let average_doc_length = self.index.values().map(|doc| doc.len()).sum::<usize>() as f32 / self.index.len() as f32;

        for (token, token_obj) in tqdm_rs::Tqdm::new(self.token_index.iter()){
            let idf = token_obj.maps.len() as f32;
            for (doc_id, freq) in token_obj.maps.iter() {
                let tf = freq.to_owned() as f32;
                let doc_len = self.index.get(doc_id).unwrap().len();
                let score = _calculate(tf, self.index.len() as f32, doc_len, average_doc_length, self.k1, self.b, idf);
                map_bm25.insert(doc_id.to_owned()+"@"+token.to_string().as_str(), score);
            }
        }
        self.map_bm25 = map_bm25;
        println!("The number of index that was freezed:{:?}", self.map_bm25.len())
    }

    fn search(&self, tokenized_quries: Vec<String>, n: usize) -> Vec<(String, f32)> {
        let mut unique_tokens = tokenized_quries.iter().collect::<Counter<_>>();
        let mut candidate_docs = HashSet::new();
        for &token in unique_tokens.keys() {
            if self.token_index.contains_key(token) {
                for doc in self.token_index.get(token).unwrap().maps.keys() {
                    candidate_docs.insert(doc.to_string());
                }
            }
        };

        let mut results = candidate_docs.iter().map(
            |doc_id| (
                doc_id.to_string(),
                tokenized_quries.iter().map(|token|
                    self.map_bm25.get(&*(doc_id.to_owned() + "@" + token.to_string().as_str())).unwrap_or(&0.0)
                ).sum::<f32>()
            )
        ).collect::<Vec<(String, f32)>>();

        results.sort_by(
            |a, b| b.1.partial_cmp(&a.1).unwrap()
        );
        results.truncate(n);
        results
    }

    fn search_instance(&self, tokenized_query: Vec<String>, n: usize) -> Vec<(String, f32)> {
        let average_doc_length = self.index.values().map(|doc| doc.len()).sum::<usize>() as f32 / self.index.len() as f32;
        let mut unique_tokens = tokenized_query.iter().collect::<Counter<_>>();
        let mut candidate_docs = HashSet::new();
        for &token in unique_tokens.keys() {
            if self.token_index.contains_key(token) {
                for doc in self.token_index.get(token).unwrap().maps.keys() {
                    candidate_docs.insert(doc.to_string());
                }
            }
        };

        let mut results: Vec<(String, f32)> = Vec::new();
        println!("length of document:{:?}, length of tokens:{:?}, total iteration:{:?}", candidate_docs.len(), unique_tokens.len(), candidate_docs.len() * unique_tokens.len());
        for doc_id in candidate_docs.iter() {
            let mut scores = Vec::new();
            for (&token, &freq) in unique_tokens.iter() {
                let mut score = 0.0;
                if self.token_index.contains_key(token) {
                    let _map = self.token_index.get(token).unwrap().maps;
                    let tf = _map.get(doc_id).unwrap_or(&0).to_owned() as f32;
                    let idf = _map.len() as f32;
                    let doc_len = self.index.get(doc_id).unwrap().len();
                    score += _calculate(tf, self.index.len() as f32, doc_len, average_doc_length, self.k1, self.b, idf) * freq as f32;
                }
                scores.push(score);
            }
            results.push((doc_id.to_string(), scores.iter().sum::<f32>()/(scores.len() as f32)));
        };
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(n);
        results
    }

    fn batch_search(&self, tokenized_queries: Vec<Vec<String>>, n: usize) -> Vec<Vec<(String, f32)>> {
        let mut scores = Vec::new();
        for tokenized_query in tqdm_rs::Tqdm::new(tokenized_queries.iter()){
            scores.push(self.search(tokenized_query.to_vec(), n));
        }
        scores
    }

    fn add_documents(&mut self, tokenized_docs: Vec<(String, Vec<String>)>){
        for (id, tokenized_doc) in tqdm_rs::Tqdm::new(tokenized_docs.iter()) {
            self.add_document(id.to_string(), tokenized_doc.to_vec());
        }
    }

    fn add_document(&mut self, id:String, tokenized_doc: Vec<String>) {
        if !self.index.contains_key(&id) {
            let mut document = Document{
                id: id.to_string(),
                maps: HashMap::new(),
            };

            for (&token, &freq) in tokenized_doc.iter().collect::<Counter<_>>().iter() {
                document.add_neighbour(token, freq as i32);
                if !self.token_index.contains_key(token){
                    let mut obj = Token { text: token.to_string(), maps: HashMap::new()};
                    obj.add_neighbour(&id, freq as i32);
                    self.token_index.insert(token.to_string(), obj);
                } else{
                    self.token_index.get_mut(token).unwrap().add_neighbour(&id, freq as i32);
                }
            }
            self.index.insert(id.to_string(), document);
        }
    }
}

#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}