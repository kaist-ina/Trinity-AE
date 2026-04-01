(seq
    (loop 0 4096 tile_n n
        (loop 0 4096 tile_k k
            (store (tensor Q1,K1,V1)
                (+
                    (* (load (tensor Q1,K1,V1) (index (fulltile) (tile n))) 1)
                    (@
                        (load (input X) (index (fulltile) (tile k)))
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
        (loop 0 1024 tile_p p
            (seq
                (store (output C_out1)
                    (rsum
                        (load (tensor C_div) (index (tile h) (fulltile) (tile p)))
                        1
                    )
                    (index (tile h) (tile p))
                )
                (store (output C_out2)
                    (rsum
                        (sqr (load (tensor C_div) (index (tile h) (fulltile) (tile p))))
                        1
                    )
                    (index (tile h) (tile p))
                )
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
)))))))))))