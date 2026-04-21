use egg::{test_fn2, test_fn_not2, *};
use std::io::BufWriter;
use std::io::Write;
use std::fs::File;
use std::path::PathBuf;
use rayon::prelude::*;
use trinity::*;
use trinity::language::{TileLang, LoopAnalysis, SHAPE_TRACKER};
use trinity::shape::{ShapeTracker, TensorShape, Dimension};
use trinity::cost::{create_fine_grained_extractor};
use egg::*;
use std::sync::Once;

pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

fn get_expressions_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("expressions")
}

// Helper function to set up shape tracker for tests
fn setup_shape_tracker(shapes: Vec<(&str, Vec<usize>)>) {
    SHAPE_TRACKER.with(|tracker| {
        let mut tracker = tracker.borrow_mut();
        *tracker = ShapeTracker::new();
        for (name, dims) in shapes {
            tracker.add_tensor(name, dims);
        }
    });
}

// Helper function to get tensor shape from an eclass
fn get_tensor_shape(egraph: &EGraph, id: Id) -> Option<TensorShape> {
    egraph[id].data.tensor_shape.clone()
}

#[test]
fn llama_extract_ffn_expressions() {
    setup_shape_tracker(vec![
        ("O2", vec![16, 4096]),
        ("WO", vec![4096, 4096]),
        ("attn_O1", vec![16, 4096]),
        ("X", vec![16, 4096]),
        ("attn_O2", vec![16, 4096]),
        ("attn_O3", vec![16]),
        ("attn_O_norm", vec![16, 4096]),
        ("WFF1a", vec![4096, 16384]),
        ("WFF1b", vec![4096, 16384]),
        ("FF1a", vec![16, 16384]),
        ("FF1b", vec![16, 16384]),
        ("FF1b_silu", vec![16, 16384]),
        ("FF1", vec![16, 16384]),
        ("FF2", vec![16, 4096]),
        ("WFF2", vec![16384, 4096]),
        ("O_FF", vec![16, 4096]),
    ]);
    let expr = "
(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor attn_O1)
                (+
                    (load (tensor attn_O1) (index (fulltile) (tile n)))
                    (@
                        (load (input O2) (index (fulltile) (tile k)))
                        (load (input WO) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 tile_n n
        (store (tensor attn_O2)
            (+
                (load (tensor attn_O1) (index (fulltile) (tile n)))
                (load (input X) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor attn_O3)
            (+
                (load (tensor attn_O3) (index (fulltile)))
                (rsum
                    (sqr (load (tensor attn_O2) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor attn_O_norm)
            (/
                (load (tensor attn_O2) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor attn_O3) (index (fulltile)))
                            4096
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 16384 tile_p p
        (loop 0 4096 tile_k k
            (store (tensor FF1a)
                (+
                    (load (tensor FF1a) (index (fulltile) (tile p)))
                    (@
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1a) (index (tile k) (tile p)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 16384 tile_p p
        (loop 0 4096 tile_k k
            (store (tensor FF1b)
                (+
                    (load (tensor FF1b) (index (fulltile) (tile p)))
                    (@
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1b) (index (tile k) (tile p)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 16384 tile_p p
        (store (tensor FF1b_silu)
            (*
                (load (tensor FF1b) (index (fulltile) (tile p)))
                (sigmoid
                    (load (tensor FF1b) (index (fulltile) (tile p)))
                )
            )
            (index (fulltile) (tile p))
        )
    )
(seq
    (loop 0 16384 tile_p p
        (store (tensor FF1)
            (*
                (load (tensor FF1a) (index (fulltile) (tile p)))
                (load (tensor FF1b_silu) (index (fulltile) (tile p)))
            )
            (index (fulltile) (tile p))
        )
    )
    (loop 0 4096 tile_n n
        (loop 0 16384 tile_p p
            (store (output FF2)
                (+
                    (load (output FF2) (index (fulltile) (tile n)))
                    (@
                        (load (tensor FF1) (index (fulltile) (tile p)))
                        (load (input WFF2) (index (tile p) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
))))))))
    ";
    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
    );

    // let all_possibilities = count_expressions_all_for_root(&runner);
    // println!("{:?}", all_possibilities);

    let expr_path = get_expressions_path();
    let semi_path = expr_path.join("semi/llama_ffn_cost6_kern5_wo_scheduler2.json");
    let output_path = expr_path.join("llama_ffn_cost6_kern5_wo_scheduler2.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 5) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }

    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, semi_path.to_str().unwrap(), usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            println!("{:?}", tile_sets);
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create(&output_path).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{}: {}", i, new_expr)  // Convert to String here
        })
        .collect::<Vec<String>>()  // Now collecting Vec<String>
        .iter()
        .for_each(|line| {
            writeln!(writer, "{}", line).expect("Failed to write to file");
        });
    
    writer.flush().expect("Failed to flush writer");
}

#[test]
fn falcon_extract_ffn_expressions() {
    setup_shape_tracker(vec![
        ("O2", vec![16, 4544]),
        ("WO", vec![4544, 4544]),
        ("attn_O1", vec![16, 4544]),
        ("X", vec![16, 4544]),
        ("attn_O2", vec![16, 4544]),
        ("attn_O3", vec![16]),
        ("attn_O_norm", vec![16, 4544]),
        ("WFF1a", vec![4544, 18176]),
        ("WFF1b", vec![4544, 18176]),
        ("FF1a", vec![16, 18176]),
        ("FF1b", vec![16, 18176]),
        ("FF1b_silu", vec![16, 18176]),
        ("FF1", vec![16, 18176]),
        ("FF2", vec![16, 4544]),
        ("WFF2", vec![18176, 4544]),
        ("O_FF", vec![16, 4544]),
    ]);
    let expr = "
(seq
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor attn_O1)
                (+
                    (load (tensor attn_O1) (index (fulltile) (tile n)))
                    (@
                        (load (input O2) (index (fulltile) (tile k)))
                        (load (input WO) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 tile_n n
        (store (tensor attn_O2)
            (+
                (load (tensor attn_O1) (index (fulltile) (tile n)))
                (load (input X) (index (fulltile) (tile n)))
            )
            (index (fulltile) (tile n))
        )
    )
(seq
    (loop 0 4544 tile_k k
        (store (tensor attn_O3)
            (+
                (load (tensor attn_O3) (index (fulltile)))
                (rsum
                    (sqr (load (tensor attn_O2) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4544 tile_k k
        (store (tensor attn_O_norm)
            (/
                (load (tensor attn_O2) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor attn_O3) (index (fulltile)))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
(seq
    (loop 0 18176 tile_p p
        (loop 0 4544 tile_k k
            (store (tensor FF1a)
                (+
                    (load (tensor FF1a) (index (fulltile) (tile p)))
                    (@
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1a) (index (tile k) (tile p)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 18176 tile_p p
        (loop 0 4544 tile_k k
            (store (tensor FF1b)
                (+
                    (load (tensor FF1b) (index (fulltile) (tile p)))
                    (@
                        (load (tensor attn_O_norm) (index (fulltile) (tile k)))
                        (load (input WFF1b) (index (tile k) (tile p)))
                    )
                )
                (index (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 18176 tile_p p
        (store (tensor FF1b_silu)
            (*
                (load (tensor FF1b) (index (fulltile) (tile p)))
                (sigmoid
                    (load (tensor FF1b) (index (fulltile) (tile p)))
                )
            )
            (index (fulltile) (tile p))
        )
    )
(seq
    (loop 0 18176 tile_p p
        (store (tensor FF1)
            (*
                (load (tensor FF1a) (index (fulltile) (tile p)))
                (load (tensor FF1b_silu) (index (fulltile) (tile p)))
            )
            (index (fulltile) (tile p))
        )
    )
    (loop 0 4544 tile_n n
        (loop 0 18176 tile_p p
            (store (output FF2)
                (+
                    (load (output FF2) (index (fulltile) (tile n)))
                    (@
                        (load (tensor FF1) (index (fulltile) (tile p)))
                        (load (input WFF2) (index (tile p) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
))))))))
    ";
    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
    );

    // let all_possibilities = count_expressions_all_for_root(&runner);
    // println!("{:?}", all_possibilities);

    let expr_path = get_expressions_path();
    let semi_path = expr_path.join("semi/falcon_ffn_cost6_kern5_wo_scheduler2.json");
    let output_path = expr_path.join("falcon_ffn_cost6_kern5_wo_scheduler2.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 5) {
        Ok(count) => println!("Saved {} expressions", count),
        Err(e) => eprintln!("Save error: {}", e),
    }

    let (expressions, tile_sets) = match list_expressions_from_semi_with_cost(&runner, semi_path.to_str().unwrap(), usize::MAX) {
        Ok((expressions, tile_sets)) => {
            println!("Loaded {} final expressions", expressions.len());
            println!("{:?}", tile_sets);
            (expressions, tile_sets)
        },
        Err(e) => {
            println!("Load error: {}", e);
            return;
        }
    };

    let file = File::create(&output_path).expect("Failed to create file");
    let mut writer = BufWriter::new(file);
    
    expressions
        .par_iter()
        .enumerate()
        .map(|(i, expr)| {
            let new_expr = postprocess_v2(expr, &tile_sets);
            format!("{}: {}", i, new_expr)  // Convert to String here
        })
        .collect::<Vec<String>>()  // Now collecting Vec<String>
        .iter()
        .for_each(|line| {
            writeln!(writer, "{}", line).expect("Failed to write to file");
        });
    
    writer.flush().expect("Failed to flush writer");
}