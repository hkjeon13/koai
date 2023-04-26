
fn sliding_text_sequence(tokens:Vec<String>, window_size:usize, stride:usize) -> Vec<Vec<String>> {
    let n_iter = ((tokens.len() - window_size) as f32 / stride as f32).ceil() as usize + 1;
    let mut outputs: Vec<Vec<String>> = Vec::with_capacity(n_iter);
    for i in 0..n_iter {
        let output = tokens[i * stride..(i * stride + window_size).min(tokens.len())].to_vec();
        outputs.push(output);
    }
    outputs
}


