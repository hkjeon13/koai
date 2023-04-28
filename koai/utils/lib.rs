extern crate pyo3;
use std::collections::HashMap;
use pyo3::prelude::*;
use serde_derive::{Serialize,Deserialize};
use std::path::Path;
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
    fn add_neighbour(&mut self, neighbour: &str) {
        if self.maps.contains_key(neighbour) {
            *self.maps.get_mut(neighbour).unwrap() += 1;
        } else {
            self.maps.insert(neighbour.to_string(), 1);
        }
    }
    fn remove_neighbour(&mut self, neighbour: &str) {
        if self.maps.contains_key(neighbour) {
            *self.maps.get_mut(neighbour).unwrap() -= 1;
        }
        if self.maps.get(neighbour).unwrap() == &0 {
            self.maps.remove(neighbour);
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
    fn add_neighbour(&mut self, neighbour: &str) {
        if self.maps.contains_key(neighbour) {
            *self.maps.get_mut(neighbour).unwrap() += 1;
        } else {
            self.maps.insert(neighbour.to_string(), 1);
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
    map_bm25: HashMap<String, HashMap<String, f32>>,
    average_length: f32,
    k1: f32,
    b: f32,
}

#[pymethods]
impl BM25 {
    #[new]
    fn new() -> Self {
        BM25 {
            index: HashMap::new(),
            token_index: HashMap::new(),
            map_bm25: HashMap::new(),
            average_length: 0.0,
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

    fn calculate_score(&self, tokenized_query: &Vec<String>, doc: &Document) -> f32 {
        let temp = HashMap::new();
        tokenized_query.iter().map(|token|self.map_bm25.get(token).unwrap_or(&temp).get(&doc.id).unwrap_or(&0.0)).sum::<f32>()
    }


    fn freeze(&mut self) {
        let temp = Token { text: "".to_string(), maps: HashMap::new()};
        let num_docs = self.index.len() as f32;
        self.average_length = self.index.iter().map(|(_, doc)| doc.len()).sum::<usize>() as f32 / num_docs;

        fn _calculate(tf: f32, num_docs:f32, doc_len: usize, average_length: f32, k1: f32, b: f32, idf: f32) -> f32 {
            (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len as f32 / average_length))) * (((num_docs as f32 - idf + 0.5) / (idf + 0.5))+1.0).ln()
        }

        self.map_bm25 = self.token_index.iter().map(
            |(token, tobj)|
                (
                    token.to_string(),
                    tobj.maps.iter().map(|(doc_id,tf)| {
                        (
                            doc_id.to_string(),
                            _calculate(
                                *tf as f32,
                                num_docs,
                                self.index.get(doc_id).unwrap().len(),
                                self.average_length,
                                self.k1,
                                self.b,
                                self.token_index.get(token).unwrap_or(&temp).maps.len() as f32
                            )
                        )
                    }).collect::<HashMap<String, f32>>()
                )

        ).collect::<HashMap<String,HashMap<String, f32>>>()


    }

    fn search(&self, tokenized_query: Vec<String>, n: i32) -> PyResult<Vec<(String, f32)>> {
        if self.map_bm25.is_empty() {
            panic!("Please freeze the index before searching(run 'freeze()' function)");
        }

        let mut result = self.index.iter().map(|(id, doc)| {
            (id.to_string(), self.calculate_score(&tokenized_query, doc))
        }).collect::<Vec<_>>();

        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(result.iter().take(n as usize).map(|x|x.to_owned()).collect::<Vec<(String, f32)>>())
    }



    fn add_documents(&mut self, tokenized_docs: Vec<(String, Vec<String>)>){
        tokenized_docs.iter().for_each(|(id, tokenized_doc)| {
            self.add_document(id.to_string(), tokenized_doc.to_vec());
        });
    }

    fn add_document(&mut self, id:String, tokenized_doc: Vec<String>) {
        if !self.index.contains_key(&id) {
            let mut document = Document{
                id: id.to_string(),
                maps: HashMap::new(),
            };
            for token in tokenized_doc {
                document.add_neighbour(&token);
                if !self.token_index.contains_key(token.as_str()){
                    let mut obj = Token { text: token.to_string(), maps: HashMap::new()};
                    obj.add_neighbour(&id);
                    self.token_index.insert(token, obj);
                } else{
                    self.token_index.get_mut(token.as_str()).unwrap().add_neighbour(&id);

                }
            }
            self.index.insert(id.to_string(), document);
        }

    }

    fn remove_documents(&mut self, ids:Vec<String>) {
        for id in ids {
            let doc = self.index.get(&id).unwrap();
            for token in doc.maps.keys() {
                self.token_index.get(token).unwrap().remove_neighbour(&id);
            }
            self.index.remove(&id);
        }
        self.freeze();
    }
}

#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BM25>()?;
    Ok(())
}