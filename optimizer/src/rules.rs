use crate::applier::*;
use crate::dependency::*;
use crate::language::LoopAnalysis;
use crate::language::TileLang;
use crate::utils::*;
use egg::{rewrite as rw, *};

#[macro_export]
macro_rules! and_all {
    ($($cond:expr),+ $(,)?) => {{
        move |egraph: &mut EGraph, id: Id, subst: &Subst| {
            true $( && $cond(egraph, id, subst) )+
        }
    }};
}
pub type EGraph = egg::EGraph<TileLang, LoopAnalysis>;

pub fn custom_rules() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![
        rw!("loop-fusion-unified";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (loop ?start ?m ?tile2 ?loop_var2 ?body2))" =>
            {
                LoopFusion {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2")
                }
            }
        ),
        rw!("loop-fusion-unified-tail";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (seq (loop ?start ?m ?tile2 ?loop_var2 ?body2) ?others))" =>
            {
                LoopFusionTail {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2"), others: var("?others")
                }
            }
        ),
    ]
}

pub fn rules() -> Vec<Rewrite<TileLang, LoopAnalysis>> {
    vec![
        // rw!("custom-reorder-expressions-for-loop-fusion";
        //     "(seq (loop ?start ?n ?tile ?loop_var ?body1)
        //      (seq ?body2
        //      (seq (loop ?start ?n ?tile ?loop_var ?body3)
        //      (seq ?body4 
        //      (seq (loop ?start ?n ?tile ?loop_var ?body5)
        //           ?body6)))))" =>
        //     "(seq (loop ?start ?n ?tile ?loop_var ?body1)
        //      (seq (loop ?start ?n ?tile ?loop_var ?body3)
        //      (seq (loop ?start ?n ?tile ?loop_var ?body5)
        //      (seq ?body2 
        //      (seq ?body4
        //           ?body6)))))"
        //     if and_all! (
        //         no_all_dependency(var("?body2"), var("?body3")),
        //         no_all_dependency(var("?body2"), var("?body5")),
        //         no_all_dependency(var("?body4"), var("?body5")),
        //         no_seq(var("?body2")),
        //         no_seq(var("?body4")),
        //         no_seq(var("?body6")),
        //         no_seq(var("?body1")),
        //         no_seq(var("?body3")),
        //         no_seq(var("?body5")),
        //     )
        // ),
        // rw!("custom-move-deadcode-outside-loop";
        //     "(loop ?start ?n ?tile ?loop_var
        //         (seq ?body1 (seq ?body2 (seq ?body3 ?body4))))"
        //     =>
        //     "(seq
        //         (loop ?start ?n ?tile ?loop_var (seq ?body2 (seq ?body3 ?body4)))
        //         (loop ?start ?n ?tile ?loop_var ?body1)
        //     )"
        //     if and_all! (
        //         no_all_dependency(var("?body1"), var("?body2")),
        //         no_all_dependency(var("?body1"), var("?body3")),
        //         no_all_dependency(var("?body1"), var("?body4")),
        //         no_seq(var("?body1")), no_seq(var("?body2")), no_seq(var("?body3")), no_seq(var("?body4")),
        //     )
        // ),
        // rw!("custom-move-deadcode-outside-loop-tail";
        //     "(seq (loop ?start ?n ?tile ?loop_var (seq ?body1 (seq ?body2 (seq ?body3 ?body4)))) ?others)"
        //     =>
        //     "(seq
        //         (loop ?start ?n ?tile ?loop_var (seq ?body2 (seq ?body3 ?body4)))
        //      (seq (loop ?start ?n ?tile ?loop_var ?body1) ?others))"
        //     if and_all! (
        //         no_all_dependency(var("?body1"), var("?body2")),
        //         no_all_dependency(var("?body1"), var("?body3")),
        //         no_all_dependency(var("?body1"), var("?body4")),
        //         no_seq(var("?body1")), no_seq(var("?body2")), no_seq(var("?body3")), no_seq(var("?body4")),
        //     )
        // ),
        // rw!("custom-many-loop-fusion3-tail";
        //     "(seq (loop ?start ?n ?tile ?loop_var ?body1)
        //      (seq (loop ?start ?n ?tile ?loop_var ?body2)
        //      (seq (loop ?start ?n ?tile ?loop_var ?body3) ?others)))" =>
        //     "(seq (loop ?start ?n ?tile ?loop_var (seq ?body1 (seq ?body2 ?body3))) ?others)"
        //     if and_all! (
        //         no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
        //         no_raw_dependency(var("?body1"), var("?body3"), var("?loop_var")),
        //         no_raw_dependency(var("?body2"), var("?body3"), var("?loop_var")),
        //     )
        // ),
        // rw!("custom-many-loop-fusion3";
        //     "(seq (loop ?start ?n ?tile ?loop_var ?body1)
        //      (seq (loop ?start ?n ?tile ?loop_var ?body2)
        //           (loop ?start ?n ?tile ?loop_var ?body3)))" =>
        //     "(loop ?start ?n ?tile ?loop_var (seq ?body1 (seq ?body2 ?body3)))"
        //     if and_all! (
        //         no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
        //         no_raw_dependency(var("?body1"), var("?body3"), var("?loop_var")),
        //         no_raw_dependency(var("?body2"), var("?body3"), var("?loop_var")),
        //     )
        // ),
        rw!("seq-comm-tail";
            "(seq ?a (seq ?b ?body))" => "(seq ?b (seq ?a ?body))"
            if and_all! (
                no_all_dependency(var("?a"), var("?b")),
                no_seq(var("?a")),
                no_seq(var("?b")),
            )
        ),
        rw!("seq-comm";
            "(seq ?a ?b)" => "(seq ?b ?a)"
            if and_all!(
                no_all_dependency(var("?a"), var("?b")),
                no_seq(var("?a")),
                no_seq(var("?b")),
            )
        ),
        rw!("loop-fusion-unified";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (loop ?start ?m ?tile2 ?loop_var2 ?body2))" =>
            {
                LoopFusion {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2")
                }
            }
        ),
        rw!("loop-fusion-unified-tail";
            "(seq (loop ?start ?n ?tile1 ?loop_var1 ?body1) (seq (loop ?start ?m ?tile2 ?loop_var2 ?body2) ?others))" =>
            {
                LoopFusionTail {
                    start: var("?start"), n: var("?n"), m: var("?m"), tile1: var("?tile1"), tile2: var("?tile2"),
                    loop_var1: var("?loop_var1"), loop_var2: var("?loop_var2"), body1: var("?body1"), body2: var("?body2"), others: var("?others")
                }
            }
        ),
        rw!("loop-fission-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (seq (loop 0 ?n ?tile_n ?loop_var ?body2) ?others))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),
        rw!("loop-fission";
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (loop 0 ?n ?tile_n ?loop_var ?body2))"
            if no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var"))
        ),
        rw!("loop-insertion1-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) (seq ?body2 ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?body2"), var("?loop_var")),
                not_same_loop(var("?body2"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion1";
            "(seq (loop 0 ?n ?tile_n ?loop_var ?body1) ?body2)" =>
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))"
            if and_all!(
                no_dependency_with_loopvar(var("?body2"), var("?loop_var")),
                not_same_loop(var("?body2"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_seq(var("?body2")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion2-tail";
            "(seq ?body1 (seq (loop 0 ?n ?tile_n ?loop_var ?body2) ?others))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2)) ?others)"
            if and_all!(
                no_dependency_with_loopvar(var("?body1"), var("?loop_var")),
                not_same_loop(var("?body1"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-insertion2";
            "(seq ?body1 (loop 0 ?n ?tile_n ?loop_var ?body2))" =>
            "(loop 0 ?n ?tile_n ?loop_var (seq ?body1 ?body2))"
            if and_all!(
                no_dependency_with_loopvar(var("?body1"), var("?loop_var")),
                not_same_loop(var("?body1"), var("?n"), var("?tile_n"), var("?loop_var")),
                no_raw_dependency(var("?body1"), var("?body2"), var("?loop_var")),
            )
        ),
        rw!("loop-deletion";
            "(loop 0 ?n ?tile_n ?loop_var ?body)" =>
            "?body"
            if and_all! (
                no_dependency_with_loopvar(var("?body"), var("?loop_var")),
                is_not_num(var("?tile_n"))
            )
        ),
        rw!("loop-comm-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq
                        (store ?a (+ (load ?a ?idx) ?val1) ?idx)
                        (store ?b (+ (load ?b ?idx) ?val2) ?idx)))
            (seq (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx) ?others))"
            =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (load ?c ?idx) (+ ?val1 ?val2)) ?idx)) ?others)"
        ),
        rw!("loop-comm";
            "(seq (loop 0 ?n ?tile_n ?loop_var (seq
                        (store ?a (+ (load ?a ?idx) ?val1) ?idx)
                        (store ?b (+ (load ?b ?idx) ?val2) ?idx)))
                (store ?c (+ (load ?a ?idx) (load ?b ?idx)) ?idx))"
            =>
            "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (load ?c ?idx) (+ ?val1 ?val2)) ?idx))"
        ),
        rw!("loop-factor-div";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) (/ ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) ?val1) ?idx))
                  (store ?b (/ (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-div-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) (/ ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) ?val1) ?idx))
                  (seq (store ?b (/ (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2"))
            )
        ),
        rw!("loop-factor-mul";
            "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) (* ?val1 ?val2)) ?idx))" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) ?val1) ?idx))
                  (store ?b (* (load ?b ?idx) ?val2) ?idx))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),
        rw!("loop-factor-mul-tail";
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) (* ?val1 ?val2)) ?idx)) ?others)" =>
            "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (load ?b ?idx) ?val1) ?idx))
                  (seq (store ?b (* (load ?b ?idx) ?val2) ?idx) ?others))"
            if and_all!(
                no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
                is_not_one(var("?val2")),
            )
        ),
        // Index bug. b의 idx가 matmul을 하기 전과 후에 달라진다.
        // rw!("loop-factor-matmul";
        //     "(loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (@ ?val1 ?val2)) ?idx))" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //           (store ?b (* (load ?b ?idx) ?val2) ?idx))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-factor-matmul-tail";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) (@ ?val1 ?val2)) ?idx)) ?others)" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //           (seq (store ?b (* (load ?b ?idx) ?val2) ?idx) ?others))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-dist-matmul-tail";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //      (seq (store ?c (* (load ?b ?idx) ?val2) ?idx2) ?others))" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx2) ?accm) (@ ?val1 ?val2)) ?idx2)) ?others)"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-dist-matmul";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //       (store ?c (* (load ?b ?idx) ?val2) ?idx2))" =>
        //     "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx2) ?accm) (@ ?val1 ?val2)) ?idx2))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-dist-div-tail";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //      (seq (store ?c (/ (load ?b ?idx) ?val2) ?idx) ?others))" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (/ ?val1 ?val2)) ?idx)) ?others)"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-dist-div";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //       (store ?c (/ (load ?b ?idx) ?val2) ?idx))" =>
        //     "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (/ ?val1 ?val2)) ?idx))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2"))
        //     )
        // ),
        // rw!("loop-dist-mul-tail";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //      (seq (store ?c (* (load ?b ?idx) ?val2) ?idx) ?others))" =>
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (* ?val1 ?val2)) ?idx)) ?others)"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2")),
        //     )
        // ),
        // rw!("loop-dist-mul";
        //     "(seq (loop 0 ?n ?tile_n ?loop_var (store ?b (+ (* (load ?b ?idx) ?accm) ?val1) ?idx))
        //       (store ?c (* (load ?b ?idx) ?val2) ?idx))" =>
        //     "(loop 0 ?n ?tile_n ?loop_var (store ?c (+ (* (load ?c ?idx) ?accm) (* ?val1 ?val2)) ?idx))"
        //     if and_all!(
        //         no_dependency_with_loopvar(var("?val2"), var("?loop_var")),
        //         is_not_one(var("?val2")),
        //     )
        // ),

        // rw!("loop-split";
        //     "(loop 0 ?end ?tile ?loop_var (store (input ?a) (+ (* (load (input ?a) ?idx) 1) ?body) ?idx))" =>
        //     { LoopSplit{
        //         end: var("?end"), tile: var("?tile"), loop_var: var("?loop_var"), a: var("?a"), idx: var("?idx"), body: var("?body"),
        //         new_tile: var("?new_tile"), new_loop_var: var("?new_loop_var"), new_a: var("?new_a"), others: var("?others"),
        //         rhs: "(seq
        //         (dloop 0 ?end ?new_tile ?new_loop_var
        //           (dloop ?new_loop_var (+ ?new_loop_var ?new_tile) ?tile ?loop_var (store (input ?new_a) (+ (* (load (input ?new_a) (index (elem ?new_loop_var) ?idx)) 1) ?body) (index (elem ?new_loop_var) ?idx))))
        //         (store (input ?a) (rsum (load (input ?new_a) (index (fulltile) ?idx)) 0) ?idx))".parse().unwrap(),
        //     }}
        //     if and_all!(
        //         has_dependency_with_loopvar(var("?body"),var("?loop_var"))
        //     )
        // ),
        // rw!("loop-split-tail";
        //     "(seq (loop 0 ?end ?tile ?loop_var (store (tensor ?a) (+ (* (load (tensor ?a) ?idx) 1) ?body) ?idx)) ?others)" =>
        //     { LoopSplitTail{
        //         end: var("?end"), tile: var("?tile"), loop_var: var("?loop_var"), a: var("?a"), idx: var("?idx"), body: var("?body"),
        //         new_tile: var("?new_tile"), new_loop_var: var("?new_loop_var"), new_a: var("?new_a"), others: var("?others"),
        //         rhs: "(seq
        //         (dloop 0 ?end ?new_tile ?new_loop_var
        //           (dloop ?new_loop_var (+ ?new_loop_var ?new_tile) ?tile ?loop_var (store (tensor ?new_a) (+ (* (load (tensor ?new_a) (index (elem ?new_loop_var) ?idx)) 1) ?body) (index (elem ?new_loop_var) ?idx))))
        //         (seq (store (tensor ?a) (rsum (load (tensor ?new_a) (index (fulltile) ?idx)) 0) ?idx) ?others))".parse().unwrap(),
        //     }}
        //     if and_all!(
        //         has_dependency_with_loopvar(var("?body"),var("?loop_var"))
        //     )
        // ),

        // algebraic transformation rules
        rw!("comm-add";  "(+ ?a ?b)"        => "(+ ?b ?a)"),
        rw!("assoc-add"; "(+ ?a (+ ?b ?c))" => "(+ (+ ?a ?b) ?c)"),
        rw!("assoc-add2"; "(+ (+ ?a ?b) ?c)" => "(+ ?a (+ ?b ?c))"),
        rw!("comm-mul";  "(* ?a ?b)"        => "(* ?b ?a)"),
        rw!("assoc-mul"; "(* ?a (* ?b ?c))" => "(* (* ?a ?b) ?c)"),
        rw!("assoc-mul2"; "(* (* ?a ?b) ?c)" => "(* ?a (* ?b ?c))"),
        rw!("assoc-matmul"; "(@ ?a (@ ?b ?c))" => "(@ (@ ?a ?b) ?c)"),
        rw!("assoc-matmul2"; "(@ (@ ?a ?b) ?c)" => "(@ ?a (@ ?b ?c))"),
        rw!("assoc-div-matmul"; "(@ (/ ?a (bcast ?b ?axis)) ?c)" => "(/ (@ ?a ?c) (bcast ?b ?axis))"),
        rw!("dist-mul-add"; "(* ?a (+ ?b ?c))"        => "(+ (* ?a ?b) (* ?a ?c))"),
        rw!("dist-mul-sub"; "(* ?a (- ?b ?c))"        => "(- (* ?a ?b) (* ?a ?c))"),
        rw!("dist-matmul-add"; "(@ ?a (+ ?b ?c))"        => "(+ (@ ?a ?b) (@ ?a ?c))"),
        rw!("dist-matmul-sub"; "(@ ?a (- ?b ?c))"        => "(- (@ ?a ?b) (@ ?a ?c))"),
        rw!("factor-mul-add"    ; "(+ (* ?a ?b) (* ?a ?c))" => "(* ?a (+ ?b ?c))"),
        rw!("factor-mul-sub"    ; "(- (* ?a ?b) (* ?a ?c))" => "(* ?a (- ?b ?c))"),
        rw!("factor-matmul-add"    ; "(+ (@ ?a ?b) (@ ?a ?c))" => "(@ ?a (+ ?b ?c))"),
        rw!("factor-matmul-sub"    ; "(- (@ ?a ?b) (@ ?a ?c))" => "(@ ?a (- ?b ?c))"),
        rw!("exp-mul"; "(* (exp ?a) (exp ?b))" => "(exp (+ ?a ?b))"),
        rw!("exp-div"; "(/ (exp ?a) (exp ?b))" => "(exp (- ?a ?b))"),
        rw!("exp0"; "(exp 0)" => "1"),
        rw!("recip-mul-div"; "(* ?x (/ 1 ?x))" => "1" if is_not_zero(var("?x"))),
        rw!("geometry-of-concat"; "(concat (concat ?x ?z 0) (concat ?y ?w 0) 1)" => "(concat (concat ?x ?y 1) (concat ?z ?w 1) 0)"),
        rw!("geometry-of-concat-inv"; "(concat (concat ?x ?y 1) (concat ?z ?w 1) 0)" => "(concat (concat ?x ?z 0) (concat ?y ?w 0) 1)"),
        rw!("operator-comm6"; "(+ (concat ?x ?z ?a) (concat ?y ?w ?a))" => "(concat (+ ?x ?y) (+ ?z ?w) ?a)"),
        rw!("operator-comm6-inv"; "(concat (+ ?x ?y) (+ ?z ?w) ?a)" => "(+ (concat ?x ?z ?a) (concat ?y ?w ?a))"),
        rw!("operator-comm7"; "(* (concat ?x ?z ?a) (concat ?y ?w ?a))" => "(concat (* ?x ?y) (* ?z ?w) ?a)"),
        rw!("operator-comm7-inv"; "(concat (* ?x ?y) (* ?z ?w) ?a)" => "(* (concat ?x ?z ?a) (concat ?y ?w ?a))"),
        rw!("concat-and-matmul0"; "(@ ?x (concat ?y ?z 1))" => "(concat (@ ?x ?y) (@ ?x ?z) 1)"),
        rw!("concat-and-matmul0-inv"; "(concat (@ ?x ?y) (@ ?x ?z) 1)" => "(@ ?x (concat ?y ?z 1))"),
        rw!("concat-and-matmul1"; "(+ (@ ?a ?b) (@ ?c ?d))" => "(@ (concat ?a ?c 1) (concat ?b ?d 0))"),
        rw!("concat-and-matmul1-inv"; "(@ (concat ?a ?c 1) (concat ?b ?d 0))" => "(+ (@ ?a ?b) (@ ?c ?d))"),
    ]
}
