//! Loop optimization using e-graphs

pub mod analysis;
pub mod applier;
pub mod cost;
pub mod dependency;
pub mod egraph_io;
pub mod extract;
pub mod extract_with_cost;
pub mod language;
pub mod optimization;
pub mod postprocess;
pub mod rules;
pub mod shape;
pub mod utils;
pub mod visualizer;

// Re-export commonly used items
pub use cost::create_extractor;
pub use egraph_io::{load_raw_egraph, save_raw_egraph};
pub use extract::{
    count_expressions_all_for_root, count_expressions_num_kernel_for_root, list_expressions_all,
    list_expressions_num_kernel,
};
pub use extract_with_cost::{
    list_expressions_from_semi_all, list_expressions_from_semi_naive,
    list_expressions_from_semi_with_cost,
};
pub use extract_with_cost::{
    list_expressions_with_target_cost, list_expressions_with_target_cost_v2,
    list_expressions_with_target_cost_v3, list_expressions_with_target_cost_v3_part1,
    list_expressions_with_target_cost_v3_part2,
};
pub use language::{Access, LoopAnalysis, LoopData, TileLang};
pub use optimization::run_until_saturated;
pub use postprocess::{postprocess, postprocess_egraph, postprocess_v2};
pub use rules::{custom_rules, rules};
pub use utils::measure_enode_proportions;
pub use visualizer::{save_egraph, save_egraph_from_egraph};
// #[cfg(feature = "serde-1")]
// pub use egraph_io::{save_egraph_binary, load_egraph_binary};

// Re-export egg items for tests
pub use egg::{test_fn2, test_fn_not2, AstSize, Extractor, RecExpr, Rewrite, Runner};

// Type aliases for convenience
pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;
pub type LoopRunner = egg::Runner<TileLang, LoopAnalysis>;
