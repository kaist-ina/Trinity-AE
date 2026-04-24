//! Optimization runner and related utilities

use crate::language::LoopAnalysis;
use crate::language::TileLang;
use egg::*;
use std::collections::HashMap;

pub fn run_until_saturated(
    expr: &str,
    rules: Vec<Rewrite<TileLang, LoopAnalysis>>,
    max_iter: usize,
) -> Runner<TileLang, LoopAnalysis> {
    let parsed_expr: RecExpr<TileLang> = expr.parse().unwrap();

    // BackoffScheduler
    let default_scheduler = BackoffScheduler::default()
        .rule_match_limit("seq-comm-tail", 5000)
        .rule_ban_length("seq-comm-tail", 1)
        .rule_match_limit("seq-comm", usize::MAX)
        .rule_ban_length("seq-comm", 0)
        .rule_match_limit("loop-fusion-unified-tail", 5000)
        .rule_ban_length("loop-fusion-unified-tail", 1)
        .rule_match_limit("loop-fusion-unified", usize::MAX)
        .rule_ban_length("loop-fusion-unified", 0);

    let runner = Runner::default()
        .with_expr(&parsed_expr)
        .with_iter_limit(max_iter)
        .with_node_limit(usize::MAX)
        .with_time_limit(std::time::Duration::from_secs(3600000))
        .with_scheduler(default_scheduler)
        .run(&rules);

    // Count edges in the egraph
    let edge_count: usize = runner
        .egraph
        .classes()
        .flat_map(|class| &class.nodes)
        .map(|node| node.len())
        .sum();

    // Count direct cycles (e-nodes with children pointing to their own e-class)
    let cycle_count: usize = runner
        .egraph
        .classes()
        .map(|class| {
            let class_id = class.id;
            class
                .nodes
                .iter()
                .filter(|node| node.children().iter().any(|child_id| *child_id == class_id))
                .count()
        })
        .sum();

    println!("📦 E-classes: {}", runner.egraph.number_of_classes());
    println!("🧱 E-nodes:   {}", runner.egraph.total_size());
    println!("🔗 Edges:     {}", edge_count);
    println!("🔄 Cycles:    {}", cycle_count);

    // Count e-nodes by type
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    for class in runner.egraph.classes() {
        for node in &class.nodes {
            let node_type = match node {
                TileLang::Loop(_) => "loop",
                TileLang::DLoop(_) => "dloop",
                TileLang::TLoop(_) => "tmp_loop",
                TileLang::Input(_) => "input",
                TileLang::Output(_) => "output",
                TileLang::Tensor(_) => "tensor",
                TileLang::Tile(_) => "tile",
                TileLang::ConstTile(_) => "const_tile",
                TileLang::FullTile => "fulltile",
                TileLang::Elem(_) => "elem",
                TileLang::Index(_) => "index",
                TileLang::Load(_) => "load",
                TileLang::Store(_) => "store",
                TileLang::Seq(_) => "seq",
                TileLang::Const(_) => "const",
                TileLang::Add(_) => "+",
                TileLang::Sub(_) => "-",
                TileLang::Mul(_) => "*",
                TileLang::Div(_) => "/",
                TileLang::Le(_) => "<=",
                TileLang::Max(_) => "max",
                TileLang::Min(_) => "min",
                TileLang::Exp(_) => "exp",
                TileLang::Matmul(_) => "@",
                TileLang::ReduceSum(_) => "rsum",
                TileLang::ReduceMin(_) => "rmin",
                TileLang::ReduceMax(_) => "rmax",
                TileLang::Sqr(_) => "sqr",
                TileLang::Sqrt(_) => "sqrt",
                TileLang::Sigmoid(_) => "sigmoid",
                TileLang::Erf(_) => "erf",
                TileLang::Abs(_) => "abs",
                TileLang::Cast(_) => "cast",
                TileLang::Concat(_) => "concat",
                TileLang::Broadcast(_) => "bcast",
                TileLang::Transpose(_) => "transpose",
                TileLang::Permute3(_) => "permute3",
                TileLang::Permute4(_) => "permute4",
                TileLang::Squeeze(_) => "squeeze",
                TileLang::Unsqueeze(_) => "unsqueeze",
                TileLang::Dummy => "dummy",
                TileLang::SLoop(_) => "sloop",
                TileLang::PLoop(_) => "ploop",
                TileLang::Num(_) => "num",
                TileLang::Var(_) => "var",
            };
            *type_counts.entry(node_type.to_string()).or_insert(0) += 1;
        }
    }

    // Print counts sorted by type name
    println!("\n📊 E-node counts by type:");
    let mut sorted_types: Vec<_> = type_counts.iter().collect();
    sorted_types.sort_by_key(|(name, _)| name.as_str());
    for (node_type, count) in sorted_types {
        println!("   {}: {}", node_type, count);
    }

    runner
}

pub fn extract_best(runner: &Runner<TileLang, LoopAnalysis>) -> Result<RecExpr<TileLang>, String> {
    let extractor = Extractor::new(&runner.egraph, AstSize);
    let root_id = runner.roots[0];
    let (_, best_expr) = extractor.find_best(root_id);
    Ok(best_expr)
}

// Add other optimization utilities
