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
fn llama_extract_rmsnorm_qkv_attn_expressions() {
    setup_shape_tracker(vec![
      ("X", vec![16, 4096]),
      ("X_norm", vec![16, 4096]),
      ("WQ", vec![4096, 4096]),
      ("WK", vec![4096, 4096]),
      ("WV", vec![4096, 4096]),
      ("Q1", vec![16, 4096]),
      ("K1", vec![16, 4096]),
      ("V1", vec![16, 4096]),
      ("Q2", vec![16, 32, 128]),
      ("K2", vec![16, 32, 128]),
      ("V2", vec![16, 32, 128]),
      ("Q", vec![32, 16, 128]),
      ("K", vec![32, 16, 128]),
      ("V", vec![32, 16, 128]),
      ("K_cache", vec![32, 1024, 128]),
      ("V_cache", vec![32, 1024, 128]),
      ("C", vec![32, 16, 1024]),
      ("C_exp", vec![32, 16, 1024]),
      ("C_sum", vec![32, 16]),
      ("C_div", vec![32, 16, 1024]),
      ("O", vec![32, 16, 128]),
      ("O1", vec![16, 32, 128]),
      ("O2", vec![16, 4096]),
      ("X2", vec![16]),
  ]);

    let expr = "
(seq
    (loop 0 4096 tile_k k
        (store (tensor X2)
            (+
                (* (load (tensor X2) (index (fulltile))) 1)
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
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
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor Q1,K1,V1)
                (+
                    (* (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                    (@
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ,WK,WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (input K_cache,V_cache)
            (load (tensor K,V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 1008 16) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C)
                (@
                    (load (tensor Q) (index (tile h) (fulltile) (fulltile)))
                    (permute3
                        (load (input K_cache) (index (tile h) (tile p) (fulltile)))
                        0 2 1
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_sum)
                (+
                    (* (load (tensor C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_div)
                (/
                    (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (tensor C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 32 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor O)
                (+
                    (* (load (tensor O) (index (tile h) (fulltile) (fulltile))) 1)
                    (@
                        (load (tensor C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4096 128 n
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
))))))))))))
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
    );

    let expr_path = get_expressions_path();
    let semi_path = expr_path.join("semi/prenorm_llama_cost6_kern1.json");
    let output_path = expr_path.join("prenorm_llama_cost6_kern1.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 1) {
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
        format!("{}: {}", i, new_expr)
    })
    .filter(|line| !line.contains("dummydata"))
    .collect::<Vec<String>>() 
    .iter()
    .for_each(|line| {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    });
    
    writer.flush().expect("Failed to flush writer");

}

#[test]
fn falcon_extract_rmsnorm_qkv_attn_expressions() {
    setup_shape_tracker(vec![
      ("X", vec![16, 4544]),
      ("X_norm", vec![16, 4544]),
      ("WQ", vec![4544, 4544]),
      ("WK", vec![4544, 4544]),
      ("WV", vec![4544, 4544]),
      ("Q1", vec![16, 4544]),
      ("K1", vec![16, 4544]),
      ("V1", vec![16, 4544]),
      ("Q2", vec![16, 71, 64]),
      ("K2", vec![16, 71, 64]),
      ("V2", vec![16, 71, 64]),
      ("Q", vec![71, 16, 64]),
      ("K", vec![71, 16, 64]),
      ("V", vec![71, 16, 64]),
      ("K_cache", vec![71, 1024, 64]),
      ("V_cache", vec![71, 1024, 64]),
      ("C", vec![71, 16, 1024]),
      ("C_exp", vec![71, 16, 1024]),
      ("C_sum", vec![71, 16]),
      ("C_div", vec![71, 16, 1024]),
      ("O", vec![71, 16, 64]),
      ("O1", vec![16, 71, 64]),
      ("O2", vec![16, 4544]),
      ("X2", vec![16]),
  ]);


    let expr = "
(seq
    (loop 0 4544 tile_k k
        (store (tensor X2)
            (+
                (* (load (tensor X2) (index (fulltile))) 1)
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
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
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor Q1,K1,V1)
                (+
                    (* (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                    (@
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ,WK,WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 64 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input K_cache,V_cache)
            (load (tensor K,V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 1008 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C)
                (@
                    (load (tensor Q) (index (tile h) (fulltile) (fulltile)))
                    (permute3
                        (load (input K_cache) (index (tile h) (tile p) (fulltile)))
                        0 2 1
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_sum)
                (+
                    (* (load (tensor C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_div)
                (/
                    (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (tensor C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor O)
                (+
                    (* (load (tensor O) (index (tile h) (fulltile) (fulltile))) 1)
                    (@
                        (load (tensor C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4544 64 n
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
))))))))))))
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        8,
    );

    let expr_path = get_expressions_path();
    let semi_path = expr_path.join("semi/prenorm_falcon_cost6_kern1.json");
    let output_path = expr_path.join("prenorm_falcon_cost6_kern1.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 1) {
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
        format!("{}: {}", i, new_expr) // String 생성
    })
    .filter(|line| !line.contains("dummydata")) // "dummydata" 포함된 건 제외
    .collect::<Vec<String>>() 
    .iter()
    .for_each(|line| {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    });
    
    writer.flush().expect("Failed to flush writer");
}

#[test]
fn prenorm_split_part1() {
    setup_shape_tracker(vec![
      ("X", vec![16, 4096]),
      ("X_norm", vec![16, 4096]),
      ("X2", vec![16]),
      ("WQ", vec![4096, 4096]),
      ("WK", vec![4096, 4096]),
      ("WV", vec![4096, 4096]),
      ("Q1", vec![16, 4096]),
      ("K1", vec![16, 4096]),
      ("V1", vec![16, 4096]),
      ("Q2", vec![16, 32, 128]),
      ("K2", vec![16, 32, 128]),
      ("V2", vec![16, 32, 128]),
      ("Q", vec![32, 16, 128]),
      ("K", vec![32, 16, 128]),
      ("V", vec![32, 16, 128]),
      ("K_cache", vec![32, 1024, 128]),
      ("V_cache", vec![32, 1024, 128]),
      ("C", vec![32, 16, 1024]),
      ("C_exp", vec![32, 16, 1024]),
      ("C_sum", vec![32, 16]),
      ("C_div", vec![32, 16, 1024]),
      ("O", vec![32, 16, 128]),
      ("O1", vec![16, 32, 128]),
      ("O2", vec![16, 4096]),
  ]);

    let expr = "
(seq
    (loop 0 4096 tile_k k
        (store (tensor X2)
            (+
                (* (load (tensor X2) (index (fulltile))) 1)
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4096 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
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
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor Q1,K1,V1)
                (+
                    (* (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                    (@
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ,WK,WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4096 128 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (output Q)
            (permute3
                (load (tensor Q2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 32 tile_h h
        (store (tensor K,V)
            (permute3
                (load (tensor K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
    (loop 0 32 tile_h h
        (store (output K_cache,V_cache)
            (load (tensor K,V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 1008 16) (fulltile))
        )
    )
    ))))))
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        8,
    );
    

    let expr_path = get_expressions_path();
    let semi_path = expr_path.join("semi/prenorm_split_part1_cost6_kern1.json");
    let output_path = expr_path.join("prenorm_split_part1_cost6_kern1.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 1) {
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
        format!("{}: {}", i, new_expr) // String 생성
    })
    .filter(|line| !line.contains("dummydata")) // "dummydata" 포함된 건 제외
    .collect::<Vec<String>>() 
    .iter()
    .for_each(|line| {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    });
    
    writer.flush().expect("Failed to flush writer");
}

#[test]
fn count_all() {
    setup_shape_tracker(vec![
      ("X", vec![16, 4096]),
      ("WQ", vec![4096, 4096]),
      ("WK", vec![4096, 4096]),
      ("WV", vec![4096, 4096]),
      ("Q1", vec![16, 4096]),
      ("K1", vec![16, 4096]),
      ("V1", vec![16, 4096]),
      ("Q2", vec![16, 32, 128]),
      ("K2", vec![16, 32, 128]),
      ("V2", vec![16, 32, 128]),
      ("Q", vec![32, 16, 128]),
      ("K", vec![32, 16, 128]),
      ("V", vec![32, 16, 128]),
      ("K_cache", vec![32, 1024, 128]),
      ("V_cache", vec![32, 1024, 128]),
      ("C", vec![32, 16, 1024]),
      ("C_exp", vec![32, 16, 1024]),
      ("C_sum", vec![32, 16]),
      ("C_div", vec![32, 16, 1024]),
      ("O", vec![32, 16, 128]),
      ("O1", vec![16, 32, 128]),
      ("O2", vec![16, 4096]),
  ]);

    let expr = "
(seq
    (loop 0 4544 tile_k k
        (store (tensor X2)
            (+
                (* (load (tensor X2) (index (fulltile))) 1)
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4544 tile_k k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
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
    (loop 0 4544 tile_n n
        (loop 0 4544 tile_k k
            (store (tensor Q1,K1,V1)
                (+
                    (* (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                    (@
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ,WK,WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
(seq
    (loop 0 4544 64 n
        (store (tensor Q2,K2,V2)
            (unsqueeze (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
            (index (fulltile) (elem n) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor Q,K,V)
            (permute3
                (load (tensor Q2,K2,V2) (index (fulltile) (tile h) (fulltile)))
                1 0 2
            )
            (index (tile h) (fulltile) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (input K_cache,V_cache)
            (load (tensor K,V) (index (tile h) (fulltile) (fulltile)))
            (index (tile h) (const_tile 1008 16) (fulltile))
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C)
                (@
                    (load (tensor Q) (index (tile h) (fulltile) (fulltile)))
                    (permute3
                        (load (input K_cache) (index (tile h) (tile p) (fulltile)))
                        0 2 1
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_sum)
                (+
                    (* (load (tensor C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_div)
                (/
                    (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (tensor C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor O)
                (+
                    (* (load (tensor O) (index (tile h) (fulltile) (fulltile))) 1)
                    (@
                        (load (tensor C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
(seq
    (loop 0 71 tile_h h
        (store (tensor O1)
            (permute3
                (load (tensor O) (index (tile h) (fulltile) (fulltile)))
                1 0 2
            )
            (index (fulltile) (tile h) (fulltile))
        )
    )
    (loop 0 4544 64 n
        (store (output O2)
            (squeeze
                (load (tensor O1) (index (fulltile) (elem n) (fulltile)))
                1
            )
            (index (fulltile) (tile n))
        )
    )
))))))))))))
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        8,
    );

    let all_possibilities = count_expressions_all_for_root(&runner);
    println!("{:?}", all_possibilities);
}

#[test]
fn attention_only() {
    setup_shape_tracker(vec![
      ("X", vec![16, 4544]),
      ("X_norm", vec![16, 4544]),
      ("WQ", vec![4544, 4544]),
      ("WK", vec![4544, 4544]),
      ("WV", vec![4544, 4544]),
      ("Q1", vec![16, 4544]),
      ("K1", vec![16, 4544]),
      ("V1", vec![16, 4544]),
      ("Q2", vec![16, 71, 64]),
      ("K2", vec![16, 71, 64]),
      ("V2", vec![16, 71, 64]),
      ("Q", vec![71, 16, 64]),
      ("K", vec![71, 16, 64]),
      ("V", vec![71, 16, 64]),
      ("K_cache", vec![71, 1024, 64]),
      ("V_cache", vec![71, 1024, 64]),
      ("C", vec![71, 16, 1024]),
      ("C_exp", vec![71, 16, 1024]),
      ("C_sum", vec![71, 16]),
      ("C_div", vec![71, 16, 1024]),
      ("O", vec![71, 16, 64]),
      ("O1", vec![16, 71, 64]),
      ("O2", vec![16, 4544]),
      ("X2", vec![16]),
  ]);


    let expr = "
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C)
                (@
                    (load (tensor Q) (index (tile h) (fulltile) (fulltile)))
                    (permute3
                        (load (input K_cache) (index (tile h) (tile p) (fulltile)))
                        0 2 1
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_exp)
                (exp (load (tensor C) (index (tile h) (fulltile) (tile p))))
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_sum)
                (+
                    (* (load (tensor C_sum) (index (tile h) (fulltile))) 1)
                    (rsum
                        (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                        2
                    )
                )
                (index (tile h) (fulltile))
            )
        )
    )
(seq
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor C_div)
                (/
                    (load (tensor C_exp) (index (tile h) (fulltile) (tile p)))
                    (bcast
                        (load (tensor C_sum) (index (tile h) (fulltile)))
                        2
                    )
                )
                (index (tile h) (fulltile) (tile p))
            )
        )
    )
    (loop 0 71 tile_h h 
        (loop 0 1024 tile_p p
            (store (tensor O)
                (+
                    (* (load (tensor O) (index (tile h) (fulltile) (fulltile))) 1)
                    (@
                        (load (tensor C_div) (index (tile h) (fulltile) (tile p)))
                        (load (input V_cache) (index (tile h) (tile p) (fulltile)))
                    )
                )
                (index (tile h) (fulltile) (fulltile))
            )
        )
    )
))))
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        10,
    );

    let expr_path = get_expressions_path();
    let semi_path = expr_path.join("semi/prenorm_attnonly_cost6_kern1.json");
    let output_path = expr_path.join("prenorm_attnonly_cost6_kern1.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 1) {
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
        format!("{}: {}", i, new_expr) // String 생성
    })
    .filter(|line| !line.contains("dummydata")) // "dummydata" 포함된 건 제외
    .collect::<Vec<String>>() 
    .iter()
    .for_each(|line| {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    });
    
    writer.flush().expect("Failed to flush writer");

}

#[test]
fn prenorm_only() {
    setup_shape_tracker(vec![
      ("X", vec![16, 4544]),
      ("X_norm", vec![16, 4544]),
      ("WQ", vec![4544, 4544]),
      ("WK", vec![4544, 4544]),
      ("WV", vec![4544, 4544]),
      ("Q1", vec![16, 4544]),
      ("K1", vec![16, 4544]),
      ("V1", vec![16, 4544]),
      ("Q2", vec![16, 71, 64]),
      ("K2", vec![16, 71, 64]),
      ("V2", vec![16, 71, 64]),
      ("Q", vec![71, 16, 64]),
      ("K", vec![71, 16, 64]),
      ("V", vec![71, 16, 64]),
      ("K_cache", vec![71, 1024, 64]),
      ("V_cache", vec![71, 1024, 64]),
      ("C", vec![71, 16, 1024]),
      ("C_exp", vec![71, 16, 1024]),
      ("C_sum", vec![71, 16]),
      ("C_div", vec![71, 16, 1024]),
      ("O", vec![71, 16, 64]),
      ("O1", vec![16, 71, 64]),
      ("O2", vec![16, 4544]),
      ("X2", vec![16]),
  ]);


    let expr = "
(seq
    (loop 0 4544 128 k
        (store (tensor X2)
            (+
                (load (tensor X2) (index (fulltile)))
                (rsum
                    (sqr (load (input X) (index (fulltile) (tile k))))
                    1
                )
            )
            (index (fulltile))
        )
    )
(seq
    (loop 0 4544 128 k
        (store (tensor X_norm)
            (/
                (load (input X) (index (fulltile) (tile k)))
                (bcast
                    (sqrt
                        (/
                            (load (tensor X2) (index (fulltile)))
                            4544
                        )
                    )
                    1
                )
            )
            (index (fulltile) (tile k))
        )
    )
    (loop 0 4544 128 n
        (loop 0 4544 128 k
            (store (tensor Q1,K1,V1)
                (+
                    (load (tensor Q1,K1,V1) (index (fulltile) (tile n)))
                    (@
                        (load (tensor X_norm) (index (fulltile) (tile k)))
                        (load (input WQ,WK,WV) (index (tile k) (tile n)))
                    )
                )
                (index (fulltile) (tile n))
            )
        )
    )
))
    ";

    let mut runner = run_until_saturated(
        expr,
        rules(),
        5,
    );
    let expr_path = get_expressions_path();
    let egraph_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("prenorm_prenormonly_egraph.dot");
    save_egraph(&runner, egraph_path.to_str().unwrap());
    return;

    let semi_path = expr_path.join("semi/prenorm_prenormonly_cost6_kern2.json");
    let output_path = expr_path.join("prenorm_prenormonly_cost6_kern2.txt");

    match list_expressions_with_target_cost_v3_part1(&runner, semi_path.to_str().unwrap(), 6, 2) {
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
        format!("{}: {}", i, new_expr) // String 생성
    })
    .filter(|line| !line.contains("dummydata")) // "dummydata" 포함된 건 제외
    .collect::<Vec<String>>() 
    .iter()
    .for_each(|line| {
        writeln!(writer, "{}", line).expect("Failed to write to file");
    });
    
    writer.flush().expect("Failed to flush writer");

}