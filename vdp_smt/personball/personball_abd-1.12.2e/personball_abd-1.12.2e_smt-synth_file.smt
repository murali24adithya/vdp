(set-logic ALL)

(define-fun t1o1 () (_ BitVec 6) (_ bv0 6))
(define-fun t1o2 () (_ BitVec 6) (_ bv1 6))
(define-fun t1o3 () (_ BitVec 6) (_ bv2 6))
(define-fun t1o4 () (_ BitVec 6) (_ bv3 6))
(define-fun t1o5 () (_ BitVec 6) (_ bv4 6))
(define-fun t1o6 () (_ BitVec 6) (_ bv5 6))
(define-fun t1o7 () (_ BitVec 6) (_ bv6 6))
(define-fun t1o8 () (_ BitVec 6) (_ bv7 6))
(define-fun t2o1 () (_ BitVec 6) (_ bv8 6))
(define-fun t2o2 () (_ BitVec 6) (_ bv9 6))
(define-fun t2o3 () (_ BitVec 6) (_ bv10 6))
(define-fun t2o4 () (_ BitVec 6) (_ bv11 6))
(define-fun t2o5 () (_ BitVec 6) (_ bv12 6))
(define-fun t2o6 () (_ BitVec 6) (_ bv13 6))
(define-fun t2o7 () (_ BitVec 6) (_ bv14 6))
(define-fun t2o8 () (_ BitVec 6) (_ bv15 6))
(define-fun t2o9 () (_ BitVec 6) (_ bv16 6))
(define-fun t3o1 () (_ BitVec 6) (_ bv17 6))
(define-fun t3o2 () (_ BitVec 6) (_ bv18 6))
(define-fun t3o3 () (_ BitVec 6) (_ bv19 6))
(define-fun t3o4 () (_ BitVec 6) (_ bv20 6))
(define-fun t3o5 () (_ BitVec 6) (_ bv21 6))
(define-fun t3o6 () (_ BitVec 6) (_ bv22 6))
(define-fun t3o7 () (_ BitVec 6) (_ bv23 6))
(define-fun t3o8 () (_ BitVec 6) (_ bv24 6))
(define-fun t3o9 () (_ BitVec 6) (_ bv25 6))
(define-fun t3o10 () (_ BitVec 6) (_ bv26 6))
(define-fun t3o11 () (_ BitVec 6) (_ bv27 6))
(define-fun t3o12 () (_ BitVec 6) (_ bv28 6))
(define-fun t3o13 () (_ BitVec 6) (_ bv29 6))
(define-fun t3o14 () (_ BitVec 6) (_ bv30 6))
(define-fun c1o1 () (_ BitVec 6) (_ bv31 6))
(define-fun c1o2 () (_ BitVec 6) (_ bv32 6))
(define-fun c2o1 () (_ BitVec 6) (_ bv33 6))
(define-fun c2o2 () (_ BitVec 6) (_ bv34 6))
(define-fun c2o3 () (_ BitVec 6) (_ bv35 6))
(define-fun c2o4 () (_ BitVec 6) (_ bv36 6))
(define-fun c2o5 () (_ BitVec 6) (_ bv37 6))
(define-fun c2o6 () (_ BitVec 6) (_ bv38 6))
(define-fun c2o7 () (_ BitVec 6) (_ bv39 6))
(define-fun c3o1 () (_ BitVec 6) (_ bv40 6))
(define-fun c3o2 () (_ BitVec 6) (_ bv41 6))
(define-fun c3o3 () (_ BitVec 6) (_ bv42 6))
(define-fun c3o4 () (_ BitVec 6) (_ bv43 6))
(define-fun c3o5 () (_ BitVec 6) (_ bv44 6))


(define-fun wine_glass () (_ BitVec 3) (_ bv0 3))
(define-fun sports_ball () (_ BitVec 3) (_ bv1 3))
(define-fun tennis_racket () (_ BitVec 3) (_ bv2 3))
(define-fun dog () (_ BitVec 3) (_ bv3 3))
(define-fun person () (_ BitVec 3) (_ bv4 3))


(define-fun labelOf((obj (_ BitVec 6))(lbl (_ BitVec 3))) Bool
(or
(and (= obj t1o1) (= lbl tennis_racket))
(and (= obj t1o2) (= lbl sports_ball))
(and (= obj t1o3) (= lbl sports_ball))
(and (= obj t1o4) (= lbl sports_ball))
(and (= obj t1o5) (= lbl sports_ball))
(and (= obj t1o6) (= lbl sports_ball))
(and (= obj t1o7) (= lbl sports_ball))
(and (= obj t1o8) (= lbl person))
(and (= obj t2o1) (= lbl sports_ball))
(and (= obj t2o2) (= lbl person))
(and (= obj t2o3) (= lbl person))
(and (= obj t2o4) (= lbl person))
(and (= obj t2o5) (= lbl person))
(and (= obj t2o6) (= lbl person))
(and (= obj t2o7) (= lbl person))
(and (= obj t2o8) (= lbl person))
(and (= obj t2o9) (= lbl person))
(and (= obj t3o1) (= lbl sports_ball))
(and (= obj t3o2) (= lbl person))
(and (= obj t3o3) (= lbl person))
(and (= obj t3o4) (= lbl person))
(and (= obj t3o5) (= lbl person))
(and (= obj t3o6) (= lbl person))
(and (= obj t3o7) (= lbl person))
(and (= obj t3o8) (= lbl person))
(and (= obj t3o9) (= lbl person))
(and (= obj t3o10) (= lbl person))
(and (= obj t3o11) (= lbl person))
(and (= obj t3o12) (= lbl person))
(and (= obj t3o13) (= lbl person))
(and (= obj t3o14) (= lbl person))
(and (= obj c1o1) (= lbl sports_ball))
(and (= obj c1o2) (= lbl dog))
(and (= obj c2o1) (= lbl wine_glass))
(and (= obj c2o2) (= lbl wine_glass))
(and (= obj c2o3) (= lbl wine_glass))
(and (= obj c2o4) (= lbl person))
(and (= obj c2o5) (= lbl person))
(and (= obj c2o6) (= lbl person))
(and (= obj c2o7) (= lbl person))
(and (= obj c3o1) (= lbl sports_ball))
(and (= obj c3o2) (= lbl person))
(and (= obj c3o3) (= lbl person))
(and (= obj c3o4) (= lbl person))
(and (= obj c3o5) (= lbl person))

))

(define-fun leftOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o3))
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o6))
(and (= x t1o1) (= y t1o7))
(and (= x t1o2) (= y t1o3))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o6))
(and (= x t1o2) (= y t1o7))
(and (= x t1o3) (= y t1o4))
(and (= x t1o3) (= y t1o6))
(and (= x t1o3) (= y t1o7))
(and (= x t1o5) (= y t1o1))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o3))
(and (= x t1o5) (= y t1o4))
(and (= x t1o5) (= y t1o6))
(and (= x t1o5) (= y t1o7))
(and (= x t1o5) (= y t1o8))
(and (= x t1o6) (= y t1o4))
(and (= x t1o6) (= y t1o7))
(and (= x t1o7) (= y t1o4))
(and (= x t1o8) (= y t1o3))
(and (= x t1o8) (= y t1o4))
(and (= x t1o8) (= y t1o6))
(and (= x t1o8) (= y t1o7))
(and (= x t2o1) (= y t2o4))
(and (= x t2o1) (= y t2o5))
(and (= x t2o1) (= y t2o8))
(and (= x t2o1) (= y t2o9))
(and (= x t2o2) (= y t2o1))
(and (= x t2o2) (= y t2o3))
(and (= x t2o2) (= y t2o4))
(and (= x t2o2) (= y t2o5))
(and (= x t2o2) (= y t2o7))
(and (= x t2o2) (= y t2o8))
(and (= x t2o2) (= y t2o9))
(and (= x t2o3) (= y t2o5))
(and (= x t2o3) (= y t2o8))
(and (= x t2o3) (= y t2o9))
(and (= x t2o4) (= y t2o8))
(and (= x t2o4) (= y t2o9))
(and (= x t2o5) (= y t2o8))
(and (= x t2o5) (= y t2o9))
(and (= x t2o6) (= y t2o1))
(and (= x t2o6) (= y t2o3))
(and (= x t2o6) (= y t2o4))
(and (= x t2o6) (= y t2o5))
(and (= x t2o6) (= y t2o8))
(and (= x t2o6) (= y t2o9))
(and (= x t2o7) (= y t2o4))
(and (= x t2o7) (= y t2o5))
(and (= x t2o7) (= y t2o8))
(and (= x t2o7) (= y t2o9))
(and (= x t2o9) (= y t2o8))
(and (= x t3o1) (= y t3o2))
(and (= x t3o1) (= y t3o3))
(and (= x t3o1) (= y t3o4))
(and (= x t3o1) (= y t3o6))
(and (= x t3o1) (= y t3o8))
(and (= x t3o1) (= y t3o9))
(and (= x t3o1) (= y t3o10))
(and (= x t3o1) (= y t3o11))
(and (= x t3o1) (= y t3o12))
(and (= x t3o1) (= y t3o13))
(and (= x t3o1) (= y t3o14))
(and (= x t3o2) (= y t3o4))
(and (= x t3o2) (= y t3o6))
(and (= x t3o2) (= y t3o8))
(and (= x t3o2) (= y t3o10))
(and (= x t3o2) (= y t3o11))
(and (= x t3o2) (= y t3o12))
(and (= x t3o2) (= y t3o14))
(and (= x t3o3) (= y t3o4))
(and (= x t3o3) (= y t3o6))
(and (= x t3o3) (= y t3o8))
(and (= x t3o3) (= y t3o10))
(and (= x t3o3) (= y t3o11))
(and (= x t3o3) (= y t3o12))
(and (= x t3o3) (= y t3o14))
(and (= x t3o5) (= y t3o2))
(and (= x t3o5) (= y t3o4))
(and (= x t3o5) (= y t3o6))
(and (= x t3o5) (= y t3o8))
(and (= x t3o5) (= y t3o9))
(and (= x t3o5) (= y t3o10))
(and (= x t3o5) (= y t3o11))
(and (= x t3o5) (= y t3o12))
(and (= x t3o5) (= y t3o13))
(and (= x t3o5) (= y t3o14))
(and (= x t3o6) (= y t3o10))
(and (= x t3o6) (= y t3o12))
(and (= x t3o6) (= y t3o14))
(and (= x t3o7) (= y t3o1))
(and (= x t3o7) (= y t3o2))
(and (= x t3o7) (= y t3o3))
(and (= x t3o7) (= y t3o4))
(and (= x t3o7) (= y t3o6))
(and (= x t3o7) (= y t3o8))
(and (= x t3o7) (= y t3o9))
(and (= x t3o7) (= y t3o10))
(and (= x t3o7) (= y t3o11))
(and (= x t3o7) (= y t3o12))
(and (= x t3o7) (= y t3o13))
(and (= x t3o7) (= y t3o14))
(and (= x t3o8) (= y t3o6))
(and (= x t3o8) (= y t3o10))
(and (= x t3o8) (= y t3o11))
(and (= x t3o8) (= y t3o12))
(and (= x t3o8) (= y t3o14))
(and (= x t3o9) (= y t3o4))
(and (= x t3o9) (= y t3o6))
(and (= x t3o9) (= y t3o8))
(and (= x t3o9) (= y t3o10))
(and (= x t3o9) (= y t3o11))
(and (= x t3o9) (= y t3o12))
(and (= x t3o9) (= y t3o14))
(and (= x t3o11) (= y t3o14))
(and (= x t3o13) (= y t3o4))
(and (= x t3o13) (= y t3o6))
(and (= x t3o13) (= y t3o8))
(and (= x t3o13) (= y t3o10))
(and (= x t3o13) (= y t3o11))
(and (= x t3o13) (= y t3o12))
(and (= x t3o13) (= y t3o14))
(and (= x c2o1) (= y c2o3))
(and (= x c2o2) (= y c2o1))
(and (= x c2o2) (= y c2o3))
(and (= x c2o2) (= y c2o4))
(and (= x c2o2) (= y c2o7))
(and (= x c2o5) (= y c2o1))
(and (= x c2o5) (= y c2o3))
(and (= x c2o5) (= y c2o4))
(and (= x c2o6) (= y c2o1))
(and (= x c2o6) (= y c2o3))
(and (= x c2o6) (= y c2o4))
(and (= x c2o6) (= y c2o7))
(and (= x c2o7) (= y c2o3))
(and (= x c3o1) (= y c3o2))
(and (= x c3o1) (= y c3o3))
(and (= x c3o2) (= y c3o3))
(and (= x c3o4) (= y c3o2))
(and (= x c3o4) (= y c3o3))
(and (= x c3o5) (= y c3o2))
(and (= x c3o5) (= y c3o3))

))

(define-fun rightOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o5))
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o5))
(and (= x t1o3) (= y t1o1))
(and (= x t1o3) (= y t1o2))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o8))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o2))
(and (= x t1o4) (= y t1o3))
(and (= x t1o4) (= y t1o5))
(and (= x t1o4) (= y t1o6))
(and (= x t1o4) (= y t1o7))
(and (= x t1o4) (= y t1o8))
(and (= x t1o6) (= y t1o1))
(and (= x t1o6) (= y t1o2))
(and (= x t1o6) (= y t1o3))
(and (= x t1o6) (= y t1o5))
(and (= x t1o6) (= y t1o8))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o2))
(and (= x t1o7) (= y t1o3))
(and (= x t1o7) (= y t1o5))
(and (= x t1o7) (= y t1o6))
(and (= x t1o7) (= y t1o8))
(and (= x t1o8) (= y t1o5))
(and (= x t2o1) (= y t2o2))
(and (= x t2o1) (= y t2o6))
(and (= x t2o3) (= y t2o2))
(and (= x t2o3) (= y t2o6))
(and (= x t2o4) (= y t2o1))
(and (= x t2o4) (= y t2o2))
(and (= x t2o4) (= y t2o6))
(and (= x t2o4) (= y t2o7))
(and (= x t2o5) (= y t2o1))
(and (= x t2o5) (= y t2o2))
(and (= x t2o5) (= y t2o3))
(and (= x t2o5) (= y t2o6))
(and (= x t2o5) (= y t2o7))
(and (= x t2o7) (= y t2o2))
(and (= x t2o8) (= y t2o1))
(and (= x t2o8) (= y t2o2))
(and (= x t2o8) (= y t2o3))
(and (= x t2o8) (= y t2o4))
(and (= x t2o8) (= y t2o5))
(and (= x t2o8) (= y t2o6))
(and (= x t2o8) (= y t2o7))
(and (= x t2o8) (= y t2o9))
(and (= x t2o9) (= y t2o1))
(and (= x t2o9) (= y t2o2))
(and (= x t2o9) (= y t2o3))
(and (= x t2o9) (= y t2o4))
(and (= x t2o9) (= y t2o5))
(and (= x t2o9) (= y t2o6))
(and (= x t2o9) (= y t2o7))
(and (= x t3o1) (= y t3o7))
(and (= x t3o2) (= y t3o1))
(and (= x t3o2) (= y t3o5))
(and (= x t3o2) (= y t3o7))
(and (= x t3o3) (= y t3o1))
(and (= x t3o3) (= y t3o7))
(and (= x t3o4) (= y t3o1))
(and (= x t3o4) (= y t3o2))
(and (= x t3o4) (= y t3o3))
(and (= x t3o4) (= y t3o5))
(and (= x t3o4) (= y t3o7))
(and (= x t3o4) (= y t3o9))
(and (= x t3o4) (= y t3o13))
(and (= x t3o6) (= y t3o1))
(and (= x t3o6) (= y t3o2))
(and (= x t3o6) (= y t3o3))
(and (= x t3o6) (= y t3o5))
(and (= x t3o6) (= y t3o7))
(and (= x t3o6) (= y t3o8))
(and (= x t3o6) (= y t3o9))
(and (= x t3o6) (= y t3o13))
(and (= x t3o8) (= y t3o1))
(and (= x t3o8) (= y t3o2))
(and (= x t3o8) (= y t3o3))
(and (= x t3o8) (= y t3o5))
(and (= x t3o8) (= y t3o7))
(and (= x t3o8) (= y t3o9))
(and (= x t3o8) (= y t3o13))
(and (= x t3o9) (= y t3o1))
(and (= x t3o9) (= y t3o5))
(and (= x t3o9) (= y t3o7))
(and (= x t3o10) (= y t3o1))
(and (= x t3o10) (= y t3o2))
(and (= x t3o10) (= y t3o3))
(and (= x t3o10) (= y t3o5))
(and (= x t3o10) (= y t3o6))
(and (= x t3o10) (= y t3o7))
(and (= x t3o10) (= y t3o8))
(and (= x t3o10) (= y t3o9))
(and (= x t3o10) (= y t3o13))
(and (= x t3o11) (= y t3o1))
(and (= x t3o11) (= y t3o2))
(and (= x t3o11) (= y t3o3))
(and (= x t3o11) (= y t3o5))
(and (= x t3o11) (= y t3o7))
(and (= x t3o11) (= y t3o8))
(and (= x t3o11) (= y t3o9))
(and (= x t3o11) (= y t3o13))
(and (= x t3o12) (= y t3o1))
(and (= x t3o12) (= y t3o2))
(and (= x t3o12) (= y t3o3))
(and (= x t3o12) (= y t3o5))
(and (= x t3o12) (= y t3o6))
(and (= x t3o12) (= y t3o7))
(and (= x t3o12) (= y t3o8))
(and (= x t3o12) (= y t3o9))
(and (= x t3o12) (= y t3o13))
(and (= x t3o13) (= y t3o1))
(and (= x t3o13) (= y t3o5))
(and (= x t3o13) (= y t3o7))
(and (= x t3o14) (= y t3o1))
(and (= x t3o14) (= y t3o2))
(and (= x t3o14) (= y t3o3))
(and (= x t3o14) (= y t3o5))
(and (= x t3o14) (= y t3o6))
(and (= x t3o14) (= y t3o7))
(and (= x t3o14) (= y t3o8))
(and (= x t3o14) (= y t3o9))
(and (= x t3o14) (= y t3o11))
(and (= x t3o14) (= y t3o13))
(and (= x c2o1) (= y c2o2))
(and (= x c2o1) (= y c2o5))
(and (= x c2o1) (= y c2o6))
(and (= x c2o3) (= y c2o1))
(and (= x c2o3) (= y c2o2))
(and (= x c2o3) (= y c2o5))
(and (= x c2o3) (= y c2o6))
(and (= x c2o3) (= y c2o7))
(and (= x c2o4) (= y c2o2))
(and (= x c2o4) (= y c2o5))
(and (= x c2o4) (= y c2o6))
(and (= x c2o7) (= y c2o2))
(and (= x c2o7) (= y c2o6))
(and (= x c3o2) (= y c3o1))
(and (= x c3o2) (= y c3o4))
(and (= x c3o2) (= y c3o5))
(and (= x c3o3) (= y c3o1))
(and (= x c3o3) (= y c3o2))
(and (= x c3o3) (= y c3o4))
(and (= x c3o3) (= y c3o5))

))

(define-fun above ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o3) (= y t1o1))
(and (= x t1o3) (= y t1o2))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o2))
(and (= x t1o4) (= y t1o3))
(and (= x t1o5) (= y t1o1))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o3))
(and (= x t1o5) (= y t1o4))
(and (= x t1o5) (= y t1o6))
(and (= x t1o5) (= y t1o7))
(and (= x t1o6) (= y t1o1))
(and (= x t1o6) (= y t1o2))
(and (= x t1o6) (= y t1o3))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o2))
(and (= x t1o7) (= y t1o3))
(and (= x t3o6) (= y t3o2))
(and (= x t3o6) (= y t3o4))
(and (= x t3o6) (= y t3o13))
(and (= x t3o8) (= y t3o2))
(and (= x t3o8) (= y t3o4))
(and (= x t3o8) (= y t3o13))
(and (= x t3o9) (= y t3o2))
(and (= x t3o9) (= y t3o4))
(and (= x t3o9) (= y t3o13))
(and (= x t3o11) (= y t3o2))
(and (= x t3o11) (= y t3o4))
(and (= x t3o11) (= y t3o13))
(and (= x t3o12) (= y t3o2))
(and (= x t3o12) (= y t3o4))
(and (= x t3o12) (= y t3o13))
(and (= x t3o14) (= y t3o2))
(and (= x t3o14) (= y t3o4))
(and (= x t3o14) (= y t3o13))
(and (= x c3o4) (= y c3o1))
(and (= x c3o5) (= y c3o1))

))

(define-fun below ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o3))
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o5))
(and (= x t1o1) (= y t1o6))
(and (= x t1o1) (= y t1o7))
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o3))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o6))
(and (= x t1o2) (= y t1o7))
(and (= x t1o3) (= y t1o4))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o6))
(and (= x t1o3) (= y t1o7))
(and (= x t1o4) (= y t1o5))
(and (= x t1o6) (= y t1o5))
(and (= x t1o7) (= y t1o5))
(and (= x t3o2) (= y t3o6))
(and (= x t3o2) (= y t3o8))
(and (= x t3o2) (= y t3o9))
(and (= x t3o2) (= y t3o11))
(and (= x t3o2) (= y t3o12))
(and (= x t3o2) (= y t3o14))
(and (= x t3o4) (= y t3o6))
(and (= x t3o4) (= y t3o8))
(and (= x t3o4) (= y t3o9))
(and (= x t3o4) (= y t3o11))
(and (= x t3o4) (= y t3o12))
(and (= x t3o4) (= y t3o14))
(and (= x t3o13) (= y t3o6))
(and (= x t3o13) (= y t3o8))
(and (= x t3o13) (= y t3o9))
(and (= x t3o13) (= y t3o11))
(and (= x t3o13) (= y t3o12))
(and (= x t3o13) (= y t3o14))
(and (= x c3o1) (= y c3o4))
(and (= x c3o1) (= y c3o5))

))

(define-fun within ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o2) (= y t1o8))
(and (= x t2o1) (= y t2o3))
(and (= x t3o1) (= y t3o5))

))



(define-fun eq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (not (= x1 x2)))


(declare-const q1 Bool)

(declare-const lbl_g1 (_ BitVec 3))
(assert (or
(= lbl_g1 wine_glass)
(= lbl_g1 sports_ball)
(= lbl_g1 tennis_racket)
(= lbl_g1 dog)
(= lbl_g1 person)
)
)
(define-fun g1 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g1)
)


(declare-const constval_wine_glass (_ BitVec 3))
(assert (= constval_wine_glass wine_glass))
(declare-const constval_sports_ball (_ BitVec 3))
(assert (= constval_sports_ball sports_ball))
(declare-const constval_tennis_racket (_ BitVec 3))
(assert (= constval_tennis_racket tennis_racket))
(declare-const constval_dog (_ BitVec 3))
(assert (= constval_dog dog))
(declare-const constval_person (_ BitVec 3))
(assert (= constval_person person))

(declare-const eq_x1_x1 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const labelOf_x1_wine_glass Bool)
(declare-const labelOf_x1_sports_ball Bool)
(declare-const labelOf_x1_tennis_racket Bool)
(declare-const labelOf_x1_dog Bool)
(declare-const labelOf_x1_person Bool)
(declare-const leftOf_x1_x1 Bool)
(declare-const rightOf_x1_x1 Bool)
(declare-const above_x1_x1 Bool)
(declare-const below_x1_x1 Bool)
(declare-const within_x1_x1 Bool)

(define-fun f ((x1 (_ BitVec 6))) Bool
(and
(=>
eq_x1_x1
(eq x1 x1)
)
(=>
neq_x1_x1
(neq x1 x1)
)
(=>
labelOf_x1_wine_glass
(labelOf x1 wine_glass)
)
(=>
labelOf_x1_sports_ball
(labelOf x1 sports_ball)
)
(=>
labelOf_x1_tennis_racket
(labelOf x1 tennis_racket)
)
(=>
labelOf_x1_dog
(labelOf x1 dog)
)
(=>
labelOf_x1_person
(labelOf x1 person)
)
(=>
leftOf_x1_x1
(leftOf x1 x1)
)
(=>
rightOf_x1_x1
(rightOf x1 x1)
)
(=>
above_x1_x1
(above x1 x1)
)
(=>
below_x1_x1
(below x1 x1)
)
(=>
within_x1_x1
(within x1 x1)
)
)

)

(declare-const c1 Bool)
(declare-const c2 Bool)
(declare-const c3 Bool)


(assert(or c1 c2 c3))
(assert (=> c1 (and (not c2) (not c3))))
(assert (=> c2 (and (not c1) (not c3))))
(assert (=> c3 (and (not c1) (not c2))))

(assert
(and
(=>
(and
(= q1 true)
)
(and
(or
(g1 t1o1)
(g1 t1o2)
(g1 t1o3)
(g1 t1o4)
(g1 t1o5)
(g1 t1o6)
(g1 t1o7)
(g1 t1o8)
)
(and
(=>
(g1 t1o1)
(f t1o1)
)
(=>
(g1 t1o2)
(f t1o2)
)
(=>
(g1 t1o3)
(f t1o3)
)
(=>
(g1 t1o4)
(f t1o4)
)
(=>
(g1 t1o5)
(f t1o5)
)
(=>
(g1 t1o6)
(f t1o6)
)
(=>
(g1 t1o7)
(f t1o7)
)
(=>
(g1 t1o8)
(f t1o8)
)
)
)
)
(=>
(and
(= q1 false)
)
(or
(and
(g1 t1o1)
(f t1o1)
)
(and
(g1 t1o2)
(f t1o2)
)
(and
(g1 t1o3)
(f t1o3)
)
(and
(g1 t1o4)
(f t1o4)
)
(and
(g1 t1o5)
(f t1o5)
)
(and
(g1 t1o6)
(f t1o6)
)
(and
(g1 t1o7)
(f t1o7)
)
(and
(g1 t1o8)
(f t1o8)
)
)
)
)
)

(assert
(and
(=>
(and
(= q1 true)
)
(and
(or
(g1 t2o1)
(g1 t2o2)
(g1 t2o3)
(g1 t2o4)
(g1 t2o5)
(g1 t2o6)
(g1 t2o7)
(g1 t2o8)
(g1 t2o9)
)
(and
(=>
(g1 t2o1)
(f t2o1)
)
(=>
(g1 t2o2)
(f t2o2)
)
(=>
(g1 t2o3)
(f t2o3)
)
(=>
(g1 t2o4)
(f t2o4)
)
(=>
(g1 t2o5)
(f t2o5)
)
(=>
(g1 t2o6)
(f t2o6)
)
(=>
(g1 t2o7)
(f t2o7)
)
(=>
(g1 t2o8)
(f t2o8)
)
(=>
(g1 t2o9)
(f t2o9)
)
)
)
)
(=>
(and
(= q1 false)
)
(or
(and
(g1 t2o1)
(f t2o1)
)
(and
(g1 t2o2)
(f t2o2)
)
(and
(g1 t2o3)
(f t2o3)
)
(and
(g1 t2o4)
(f t2o4)
)
(and
(g1 t2o5)
(f t2o5)
)
(and
(g1 t2o6)
(f t2o6)
)
(and
(g1 t2o7)
(f t2o7)
)
(and
(g1 t2o8)
(f t2o8)
)
(and
(g1 t2o9)
(f t2o9)
)
)
)
)
)

(assert
(and
(=>
(and
(= q1 true)
)
(and
(or
(g1 t3o1)
(g1 t3o2)
(g1 t3o3)
(g1 t3o4)
(g1 t3o5)
(g1 t3o6)
(g1 t3o7)
(g1 t3o8)
(g1 t3o9)
(g1 t3o10)
(g1 t3o11)
(g1 t3o12)
(g1 t3o13)
(g1 t3o14)
)
(and
(=>
(g1 t3o1)
(f t3o1)
)
(=>
(g1 t3o2)
(f t3o2)
)
(=>
(g1 t3o3)
(f t3o3)
)
(=>
(g1 t3o4)
(f t3o4)
)
(=>
(g1 t3o5)
(f t3o5)
)
(=>
(g1 t3o6)
(f t3o6)
)
(=>
(g1 t3o7)
(f t3o7)
)
(=>
(g1 t3o8)
(f t3o8)
)
(=>
(g1 t3o9)
(f t3o9)
)
(=>
(g1 t3o10)
(f t3o10)
)
(=>
(g1 t3o11)
(f t3o11)
)
(=>
(g1 t3o12)
(f t3o12)
)
(=>
(g1 t3o13)
(f t3o13)
)
(=>
(g1 t3o14)
(f t3o14)
)
)
)
)
(=>
(and
(= q1 false)
)
(or
(and
(g1 t3o1)
(f t3o1)
)
(and
(g1 t3o2)
(f t3o2)
)
(and
(g1 t3o3)
(f t3o3)
)
(and
(g1 t3o4)
(f t3o4)
)
(and
(g1 t3o5)
(f t3o5)
)
(and
(g1 t3o6)
(f t3o6)
)
(and
(g1 t3o7)
(f t3o7)
)
(and
(g1 t3o8)
(f t3o8)
)
(and
(g1 t3o9)
(f t3o9)
)
(and
(g1 t3o10)
(f t3o10)
)
(and
(g1 t3o11)
(f t3o11)
)
(and
(g1 t3o12)
(f t3o12)
)
(and
(g1 t3o13)
(f t3o13)
)
(and
(g1 t3o14)
(f t3o14)
)
)
)
)
)

(assert
(=>
(= c1 true)
(and
(=>
(and
(= q1 true)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
)
(and
(=>
(g1 c1o1)
(f c1o1)
)
(=>
(g1 c1o2)
(f c1o2)
)
)
)
)
(=>
(and
(= q1 false)
)
(or
(and
(g1 c1o1)
(f c1o1)
)
(and
(g1 c1o2)
(f c1o2)
)
)
)
)
)
)

(assert
(=>
(= c1 false)
(and
(=>
(and
(= q1 true)
)
(or
(and
(g1 c1o1)
(not
(f c1o1)
)
)
(and
(g1 c1o2)
(not
(f c1o2)
)
)
)
)
(=>
(and
(= q1 false)
)
(and
(=>
(g1 c1o1)
(not
(f c1o1)
)
)
(=>
(g1 c1o2)
(not
(f c1o2)
)
)
)
)
)
)
)

(assert
(=>
(= c2 true)
(and
(=>
(and
(= q1 true)
)
(and
(or
(g1 c2o1)
(g1 c2o2)
(g1 c2o3)
(g1 c2o4)
(g1 c2o5)
(g1 c2o6)
(g1 c2o7)
)
(and
(=>
(g1 c2o1)
(f c2o1)
)
(=>
(g1 c2o2)
(f c2o2)
)
(=>
(g1 c2o3)
(f c2o3)
)
(=>
(g1 c2o4)
(f c2o4)
)
(=>
(g1 c2o5)
(f c2o5)
)
(=>
(g1 c2o6)
(f c2o6)
)
(=>
(g1 c2o7)
(f c2o7)
)
)
)
)
(=>
(and
(= q1 false)
)
(or
(and
(g1 c2o1)
(f c2o1)
)
(and
(g1 c2o2)
(f c2o2)
)
(and
(g1 c2o3)
(f c2o3)
)
(and
(g1 c2o4)
(f c2o4)
)
(and
(g1 c2o5)
(f c2o5)
)
(and
(g1 c2o6)
(f c2o6)
)
(and
(g1 c2o7)
(f c2o7)
)
)
)
)
)
)

(assert
(=>
(= c2 false)
(and
(=>
(and
(= q1 true)
)
(or
(and
(g1 c2o1)
(not
(f c2o1)
)
)
(and
(g1 c2o2)
(not
(f c2o2)
)
)
(and
(g1 c2o3)
(not
(f c2o3)
)
)
(and
(g1 c2o4)
(not
(f c2o4)
)
)
(and
(g1 c2o5)
(not
(f c2o5)
)
)
(and
(g1 c2o6)
(not
(f c2o6)
)
)
(and
(g1 c2o7)
(not
(f c2o7)
)
)
)
)
(=>
(and
(= q1 false)
)
(and
(=>
(g1 c2o1)
(not
(f c2o1)
)
)
(=>
(g1 c2o2)
(not
(f c2o2)
)
)
(=>
(g1 c2o3)
(not
(f c2o3)
)
)
(=>
(g1 c2o4)
(not
(f c2o4)
)
)
(=>
(g1 c2o5)
(not
(f c2o5)
)
)
(=>
(g1 c2o6)
(not
(f c2o6)
)
)
(=>
(g1 c2o7)
(not
(f c2o7)
)
)
)
)
)
)
)

(assert
(=>
(= c3 true)
(and
(=>
(and
(= q1 true)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
(g1 c3o3)
(g1 c3o4)
(g1 c3o5)
)
(and
(=>
(g1 c3o1)
(f c3o1)
)
(=>
(g1 c3o2)
(f c3o2)
)
(=>
(g1 c3o3)
(f c3o3)
)
(=>
(g1 c3o4)
(f c3o4)
)
(=>
(g1 c3o5)
(f c3o5)
)
)
)
)
(=>
(and
(= q1 false)
)
(or
(and
(g1 c3o1)
(f c3o1)
)
(and
(g1 c3o2)
(f c3o2)
)
(and
(g1 c3o3)
(f c3o3)
)
(and
(g1 c3o4)
(f c3o4)
)
(and
(g1 c3o5)
(f c3o5)
)
)
)
)
)
)

(assert
(=>
(= c3 false)
(and
(=>
(and
(= q1 true)
)
(or
(and
(g1 c3o1)
(not
(f c3o1)
)
)
(and
(g1 c3o2)
(not
(f c3o2)
)
)
(and
(g1 c3o3)
(not
(f c3o3)
)
)
(and
(g1 c3o4)
(not
(f c3o4)
)
)
(and
(g1 c3o5)
(not
(f c3o5)
)
)
)
)
(=>
(and
(= q1 false)
)
(and
(=>
(g1 c3o1)
(not
(f c3o1)
)
)
(=>
(g1 c3o2)
(not
(f c3o2)
)
)
(=>
(g1 c3o3)
(not
(f c3o3)
)
)
(=>
(g1 c3o4)
(not
(f c3o4)
)
)
(=>
(g1 c3o5)
(not
(f c3o5)
)
)
)
)
)
)
)


(check-sat)
(get-model)