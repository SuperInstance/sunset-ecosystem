use std::io::{self, Read};

use jepa_kernel::jepa_forward_batch;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let input = if args.len() > 1 {
        std::fs::read_to_string(&args[1]).expect("Failed to read input file")
    } else {
        let mut buf = String::new();
        io::stdin().read_to_string(&mut buf).expect("Failed to read stdin");
        buf
    };

    let parsed: serde_json::Value = serde_json::from_str(&input).expect("Invalid JSON");

    let n = parsed["n"].as_u64().expect("Missing 'n'") as usize;
    let x: Vec<f32> = serde_json::from_value(parsed["x"].clone()).expect("Invalid 'x'");
    assert_eq!(x.len(), 64, "x must be 64 floats");

    let w1: Vec<f32> = serde_json::from_value(parsed["w1"].clone()).expect("Invalid 'w1'");
    let w2: Vec<f32> = serde_json::from_value(parsed["w2"].clone()).expect("Invalid 'w2'");
    let w3: Vec<f32> = serde_json::from_value(parsed["w3"].clone()).expect("Invalid 'w3'");
    let b1: Vec<f32> = serde_json::from_value(parsed["b1"].clone()).expect("Invalid 'b1'");
    let b2: Vec<f32> = serde_json::from_value(parsed["b2"].clone()).expect("Invalid 'b2'");
    let b3: Vec<f32> = serde_json::from_value(parsed["b3"].clone()).expect("Invalid 'b3'");

    assert_eq!(w1.len(), n * 64 * 32, "w1 length mismatch");
    assert_eq!(w2.len(), n * 32 * 16, "w2 length mismatch");
    assert_eq!(w3.len(), n * 16 * 16, "w3 length mismatch");
    assert_eq!(b1.len(), n * 32, "b1 length mismatch");
    assert_eq!(b2.len(), n * 16, "b2 length mismatch");
    assert_eq!(b3.len(), n * 16, "b3 length mismatch");

    let mut out = vec![0.0f32; n * 16];

    jepa_forward_batch(
        x.as_ptr(),
        w1.as_ptr(), w2.as_ptr(), w3.as_ptr(),
        b1.as_ptr(), b2.as_ptr(), b3.as_ptr(),
        n,
        out.as_mut_ptr(),
    );

    let result = serde_json::json!({ "latents": out });
    println!("{}", serde_json::to_string(&result).unwrap());
}
