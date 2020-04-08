(set-logic ALL)

(define-fun t1o1 () (_ BitVec 6) (_ bv0 6))
(define-fun t1o2 () (_ BitVec 6) (_ bv1 6))
(define-fun t1o3 () (_ BitVec 6) (_ bv2 6))
(define-fun t1o4 () (_ BitVec 6) (_ bv3 6))
(define-fun t1o5 () (_ BitVec 6) (_ bv4 6))
(define-fun t1o6 () (_ BitVec 6) (_ bv5 6))
(define-fun t1o7 () (_ BitVec 6) (_ bv6 6))
(define-fun t1o8 () (_ BitVec 6) (_ bv7 6))
(define-fun t1o9 () (_ BitVec 6) (_ bv8 6))
(define-fun t1o10 () (_ BitVec 6) (_ bv9 6))
(define-fun t1o11 () (_ BitVec 6) (_ bv10 6))
(define-fun t1o12 () (_ BitVec 6) (_ bv11 6))
(define-fun t1o13 () (_ BitVec 6) (_ bv12 6))
(define-fun t1o14 () (_ BitVec 6) (_ bv13 6))
(define-fun t1o15 () (_ BitVec 6) (_ bv14 6))
(define-fun t1o16 () (_ BitVec 6) (_ bv15 6))
(define-fun t2o1 () (_ BitVec 6) (_ bv16 6))
(define-fun t2o2 () (_ BitVec 6) (_ bv17 6))
(define-fun t2o3 () (_ BitVec 6) (_ bv18 6))
(define-fun t2o4 () (_ BitVec 6) (_ bv19 6))
(define-fun t2o5 () (_ BitVec 6) (_ bv20 6))
(define-fun t2o6 () (_ BitVec 6) (_ bv21 6))
(define-fun t2o7 () (_ BitVec 6) (_ bv22 6))
(define-fun t2o8 () (_ BitVec 6) (_ bv23 6))
(define-fun t2o9 () (_ BitVec 6) (_ bv24 6))
(define-fun t2o10 () (_ BitVec 6) (_ bv25 6))
(define-fun t2o11 () (_ BitVec 6) (_ bv26 6))
(define-fun t2o12 () (_ BitVec 6) (_ bv27 6))
(define-fun t2o13 () (_ BitVec 6) (_ bv28 6))
(define-fun t2o14 () (_ BitVec 6) (_ bv29 6))
(define-fun t2o15 () (_ BitVec 6) (_ bv30 6))
(define-fun t3o1 () (_ BitVec 6) (_ bv31 6))
(define-fun t3o2 () (_ BitVec 6) (_ bv32 6))
(define-fun c1o1 () (_ BitVec 6) (_ bv33 6))
(define-fun c1o2 () (_ BitVec 6) (_ bv34 6))
(define-fun c1o3 () (_ BitVec 6) (_ bv35 6))
(define-fun c1o4 () (_ BitVec 6) (_ bv36 6))
(define-fun c1o5 () (_ BitVec 6) (_ bv37 6))
(define-fun c2o1 () (_ BitVec 6) (_ bv38 6))
(define-fun c2o2 () (_ BitVec 6) (_ bv39 6))
(define-fun c3o1 () (_ BitVec 6) (_ bv40 6))
(define-fun c3o2 () (_ BitVec 6) (_ bv41 6))
(define-fun c3o3 () (_ BitVec 6) (_ bv42 6))
(define-fun c3o4 () (_ BitVec 6) (_ bv43 6))


(define-fun person () (_ BitVec 3) (_ bv0 3))
(define-fun car () (_ BitVec 3) (_ bv1 3))
(define-fun bus () (_ BitVec 3) (_ bv2 3))
(define-fun backpack () (_ BitVec 3) (_ bv3 3))
(define-fun handbag () (_ BitVec 3) (_ bv4 3))


(define-fun labelOf((obj (_ BitVec 6))(lbl (_ BitVec 3))) Bool
(or
(and (= obj t1o1) (= lbl backpack))
(and (= obj t1o2) (= lbl bus))
(and (= obj t1o3) (= lbl car))
(and (= obj t1o4) (= lbl person))
(and (= obj t1o5) (= lbl person))
(and (= obj t1o6) (= lbl person))
(and (= obj t1o7) (= lbl person))
(and (= obj t1o8) (= lbl person))
(and (= obj t1o9) (= lbl person))
(and (= obj t1o10) (= lbl person))
(and (= obj t1o11) (= lbl person))
(and (= obj t1o12) (= lbl person))
(and (= obj t1o13) (= lbl person))
(and (= obj t1o14) (= lbl person))
(and (= obj t1o15) (= lbl person))
(and (= obj t1o16) (= lbl person))
(and (= obj t2o1) (= lbl handbag))
(and (= obj t2o2) (= lbl bus))
(and (= obj t2o3) (= lbl bus))
(and (= obj t2o4) (= lbl person))
(and (= obj t2o5) (= lbl person))
(and (= obj t2o6) (= lbl person))
(and (= obj t2o7) (= lbl person))
(and (= obj t2o8) (= lbl person))
(and (= obj t2o9) (= lbl person))
(and (= obj t2o10) (= lbl person))
(and (= obj t2o11) (= lbl person))
(and (= obj t2o12) (= lbl person))
(and (= obj t2o13) (= lbl person))
(and (= obj t2o14) (= lbl person))
(and (= obj t2o15) (= lbl person))
(and (= obj t3o1) (= lbl bus))
(and (= obj t3o2) (= lbl person))
(and (= obj c1o1) (= lbl bus))
(and (= obj c1o2) (= lbl bus))
(and (= obj c1o3) (= lbl person))
(and (= obj c1o4) (= lbl person))
(and (= obj c1o5) (= lbl person))
(and (= obj c2o1) (= lbl bus))
(and (= obj c2o2) (= lbl car))
(and (= obj c3o1) (= lbl bus))
(and (= obj c3o2) (= lbl car))
(and (= obj c3o3) (= lbl car))
(and (= obj c3o4) (= lbl person))

))

(define-fun leftOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o3))
(and (= x t1o1) (= y t1o6))
(and (= x t1o1) (= y t1o7))
(and (= x t1o1) (= y t1o10))
(and (= x t1o1) (= y t1o11))
(and (= x t1o1) (= y t1o12))
(and (= x t1o1) (= y t1o14))
(and (= x t1o1) (= y t1o15))
(and (= x t1o2) (= y t1o3))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o2))
(and (= x t1o4) (= y t1o3))
(and (= x t1o4) (= y t1o5))
(and (= x t1o4) (= y t1o6))
(and (= x t1o4) (= y t1o7))
(and (= x t1o4) (= y t1o9))
(and (= x t1o4) (= y t1o10))
(and (= x t1o4) (= y t1o11))
(and (= x t1o4) (= y t1o12))
(and (= x t1o4) (= y t1o13))
(and (= x t1o4) (= y t1o14))
(and (= x t1o4) (= y t1o15))
(and (= x t1o4) (= y t1o16))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o3))
(and (= x t1o5) (= y t1o6))
(and (= x t1o5) (= y t1o7))
(and (= x t1o5) (= y t1o10))
(and (= x t1o5) (= y t1o11))
(and (= x t1o5) (= y t1o12))
(and (= x t1o5) (= y t1o14))
(and (= x t1o5) (= y t1o15))
(and (= x t1o6) (= y t1o3))
(and (= x t1o6) (= y t1o7))
(and (= x t1o6) (= y t1o12))
(and (= x t1o7) (= y t1o3))
(and (= x t1o8) (= y t1o1))
(and (= x t1o8) (= y t1o2))
(and (= x t1o8) (= y t1o3))
(and (= x t1o8) (= y t1o5))
(and (= x t1o8) (= y t1o6))
(and (= x t1o8) (= y t1o7))
(and (= x t1o8) (= y t1o9))
(and (= x t1o8) (= y t1o10))
(and (= x t1o8) (= y t1o11))
(and (= x t1o8) (= y t1o12))
(and (= x t1o8) (= y t1o14))
(and (= x t1o8) (= y t1o15))
(and (= x t1o8) (= y t1o16))
(and (= x t1o9) (= y t1o1))
(and (= x t1o9) (= y t1o2))
(and (= x t1o9) (= y t1o3))
(and (= x t1o9) (= y t1o6))
(and (= x t1o9) (= y t1o7))
(and (= x t1o9) (= y t1o10))
(and (= x t1o9) (= y t1o11))
(and (= x t1o9) (= y t1o12))
(and (= x t1o9) (= y t1o14))
(and (= x t1o9) (= y t1o15))
(and (= x t1o9) (= y t1o16))
(and (= x t1o10) (= y t1o2))
(and (= x t1o10) (= y t1o3))
(and (= x t1o10) (= y t1o6))
(and (= x t1o10) (= y t1o7))
(and (= x t1o10) (= y t1o12))
(and (= x t1o11) (= y t1o2))
(and (= x t1o11) (= y t1o3))
(and (= x t1o11) (= y t1o7))
(and (= x t1o11) (= y t1o12))
(and (= x t1o12) (= y t1o3))
(and (= x t1o12) (= y t1o7))
(and (= x t1o13) (= y t1o1))
(and (= x t1o13) (= y t1o2))
(and (= x t1o13) (= y t1o3))
(and (= x t1o13) (= y t1o5))
(and (= x t1o13) (= y t1o6))
(and (= x t1o13) (= y t1o7))
(and (= x t1o13) (= y t1o10))
(and (= x t1o13) (= y t1o11))
(and (= x t1o13) (= y t1o12))
(and (= x t1o13) (= y t1o14))
(and (= x t1o13) (= y t1o15))
(and (= x t1o13) (= y t1o16))
(and (= x t1o14) (= y t1o2))
(and (= x t1o14) (= y t1o3))
(and (= x t1o14) (= y t1o6))
(and (= x t1o14) (= y t1o7))
(and (= x t1o14) (= y t1o11))
(and (= x t1o14) (= y t1o12))
(and (= x t1o14) (= y t1o15))
(and (= x t1o15) (= y t1o2))
(and (= x t1o15) (= y t1o3))
(and (= x t1o15) (= y t1o6))
(and (= x t1o15) (= y t1o7))
(and (= x t1o15) (= y t1o12))
(and (= x t1o16) (= y t1o2))
(and (= x t1o16) (= y t1o3))
(and (= x t1o16) (= y t1o6))
(and (= x t1o16) (= y t1o7))
(and (= x t1o16) (= y t1o10))
(and (= x t1o16) (= y t1o11))
(and (= x t1o16) (= y t1o12))
(and (= x t1o16) (= y t1o14))
(and (= x t1o16) (= y t1o15))
(and (= x t2o1) (= y t2o4))
(and (= x t2o1) (= y t2o5))
(and (= x t2o1) (= y t2o8))
(and (= x t2o1) (= y t2o9))
(and (= x t2o1) (= y t2o10))
(and (= x t2o1) (= y t2o12))
(and (= x t2o1) (= y t2o13))
(and (= x t2o1) (= y t2o14))
(and (= x t2o2) (= y t2o1))
(and (= x t2o2) (= y t2o3))
(and (= x t2o2) (= y t2o4))
(and (= x t2o2) (= y t2o5))
(and (= x t2o2) (= y t2o6))
(and (= x t2o2) (= y t2o7))
(and (= x t2o2) (= y t2o8))
(and (= x t2o2) (= y t2o9))
(and (= x t2o2) (= y t2o10))
(and (= x t2o2) (= y t2o11))
(and (= x t2o2) (= y t2o12))
(and (= x t2o2) (= y t2o13))
(and (= x t2o2) (= y t2o14))
(and (= x t2o2) (= y t2o15))
(and (= x t2o4) (= y t2o9))
(and (= x t2o4) (= y t2o10))
(and (= x t2o4) (= y t2o13))
(and (= x t2o4) (= y t2o14))
(and (= x t2o5) (= y t2o8))
(and (= x t2o5) (= y t2o9))
(and (= x t2o5) (= y t2o10))
(and (= x t2o5) (= y t2o13))
(and (= x t2o5) (= y t2o14))
(and (= x t2o6) (= y t2o1))
(and (= x t2o6) (= y t2o3))
(and (= x t2o6) (= y t2o4))
(and (= x t2o6) (= y t2o5))
(and (= x t2o6) (= y t2o8))
(and (= x t2o6) (= y t2o9))
(and (= x t2o6) (= y t2o10))
(and (= x t2o6) (= y t2o12))
(and (= x t2o6) (= y t2o13))
(and (= x t2o6) (= y t2o14))
(and (= x t2o6) (= y t2o15))
(and (= x t2o7) (= y t2o4))
(and (= x t2o7) (= y t2o5))
(and (= x t2o7) (= y t2o8))
(and (= x t2o7) (= y t2o9))
(and (= x t2o7) (= y t2o10))
(and (= x t2o7) (= y t2o12))
(and (= x t2o7) (= y t2o13))
(and (= x t2o7) (= y t2o14))
(and (= x t2o8) (= y t2o10))
(and (= x t2o8) (= y t2o13))
(and (= x t2o8) (= y t2o14))
(and (= x t2o9) (= y t2o10))
(and (= x t2o9) (= y t2o13))
(and (= x t2o9) (= y t2o14))
(and (= x t2o10) (= y t2o13))
(and (= x t2o10) (= y t2o14))
(and (= x t2o11) (= y t2o1))
(and (= x t2o11) (= y t2o3))
(and (= x t2o11) (= y t2o4))
(and (= x t2o11) (= y t2o5))
(and (= x t2o11) (= y t2o7))
(and (= x t2o11) (= y t2o8))
(and (= x t2o11) (= y t2o9))
(and (= x t2o11) (= y t2o10))
(and (= x t2o11) (= y t2o12))
(and (= x t2o11) (= y t2o13))
(and (= x t2o11) (= y t2o14))
(and (= x t2o11) (= y t2o15))
(and (= x t2o12) (= y t2o8))
(and (= x t2o12) (= y t2o9))
(and (= x t2o12) (= y t2o10))
(and (= x t2o12) (= y t2o13))
(and (= x t2o12) (= y t2o14))
(and (= x t2o13) (= y t2o14))
(and (= x t2o15) (= y t2o4))
(and (= x t2o15) (= y t2o5))
(and (= x t2o15) (= y t2o8))
(and (= x t2o15) (= y t2o9))
(and (= x t2o15) (= y t2o10))
(and (= x t2o15) (= y t2o12))
(and (= x t2o15) (= y t2o13))
(and (= x t2o15) (= y t2o14))
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o4))
(and (= x c1o2) (= y c1o5))
(and (= x c1o3) (= y c1o1))
(and (= x c1o3) (= y c1o4))
(and (= x c1o3) (= y c1o5))
(and (= x c1o4) (= y c1o1))
(and (= x c2o1) (= y c2o2))
(and (= x c3o2) (= y c3o4))
(and (= x c3o3) (= y c3o1))
(and (= x c3o3) (= y c3o2))
(and (= x c3o3) (= y c3o4))

))

(define-fun rightOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o8))
(and (= x t1o1) (= y t1o9))
(and (= x t1o1) (= y t1o13))
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o8))
(and (= x t1o2) (= y t1o9))
(and (= x t1o2) (= y t1o10))
(and (= x t1o2) (= y t1o11))
(and (= x t1o2) (= y t1o13))
(and (= x t1o2) (= y t1o14))
(and (= x t1o2) (= y t1o15))
(and (= x t1o2) (= y t1o16))
(and (= x t1o3) (= y t1o1))
(and (= x t1o3) (= y t1o2))
(and (= x t1o3) (= y t1o4))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o6))
(and (= x t1o3) (= y t1o7))
(and (= x t1o3) (= y t1o8))
(and (= x t1o3) (= y t1o9))
(and (= x t1o3) (= y t1o10))
(and (= x t1o3) (= y t1o11))
(and (= x t1o3) (= y t1o12))
(and (= x t1o3) (= y t1o13))
(and (= x t1o3) (= y t1o14))
(and (= x t1o3) (= y t1o15))
(and (= x t1o3) (= y t1o16))
(and (= x t1o5) (= y t1o4))
(and (= x t1o5) (= y t1o8))
(and (= x t1o5) (= y t1o13))
(and (= x t1o6) (= y t1o1))
(and (= x t1o6) (= y t1o4))
(and (= x t1o6) (= y t1o5))
(and (= x t1o6) (= y t1o8))
(and (= x t1o6) (= y t1o9))
(and (= x t1o6) (= y t1o10))
(and (= x t1o6) (= y t1o13))
(and (= x t1o6) (= y t1o14))
(and (= x t1o6) (= y t1o15))
(and (= x t1o6) (= y t1o16))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o4))
(and (= x t1o7) (= y t1o5))
(and (= x t1o7) (= y t1o6))
(and (= x t1o7) (= y t1o8))
(and (= x t1o7) (= y t1o9))
(and (= x t1o7) (= y t1o10))
(and (= x t1o7) (= y t1o11))
(and (= x t1o7) (= y t1o12))
(and (= x t1o7) (= y t1o13))
(and (= x t1o7) (= y t1o14))
(and (= x t1o7) (= y t1o15))
(and (= x t1o7) (= y t1o16))
(and (= x t1o9) (= y t1o4))
(and (= x t1o9) (= y t1o8))
(and (= x t1o10) (= y t1o1))
(and (= x t1o10) (= y t1o4))
(and (= x t1o10) (= y t1o5))
(and (= x t1o10) (= y t1o8))
(and (= x t1o10) (= y t1o9))
(and (= x t1o10) (= y t1o13))
(and (= x t1o10) (= y t1o16))
(and (= x t1o11) (= y t1o1))
(and (= x t1o11) (= y t1o4))
(and (= x t1o11) (= y t1o5))
(and (= x t1o11) (= y t1o8))
(and (= x t1o11) (= y t1o9))
(and (= x t1o11) (= y t1o13))
(and (= x t1o11) (= y t1o14))
(and (= x t1o11) (= y t1o16))
(and (= x t1o12) (= y t1o1))
(and (= x t1o12) (= y t1o4))
(and (= x t1o12) (= y t1o5))
(and (= x t1o12) (= y t1o6))
(and (= x t1o12) (= y t1o8))
(and (= x t1o12) (= y t1o9))
(and (= x t1o12) (= y t1o10))
(and (= x t1o12) (= y t1o11))
(and (= x t1o12) (= y t1o13))
(and (= x t1o12) (= y t1o14))
(and (= x t1o12) (= y t1o15))
(and (= x t1o12) (= y t1o16))
(and (= x t1o13) (= y t1o4))
(and (= x t1o14) (= y t1o1))
(and (= x t1o14) (= y t1o4))
(and (= x t1o14) (= y t1o5))
(and (= x t1o14) (= y t1o8))
(and (= x t1o14) (= y t1o9))
(and (= x t1o14) (= y t1o13))
(and (= x t1o14) (= y t1o16))
(and (= x t1o15) (= y t1o1))
(and (= x t1o15) (= y t1o4))
(and (= x t1o15) (= y t1o5))
(and (= x t1o15) (= y t1o8))
(and (= x t1o15) (= y t1o9))
(and (= x t1o15) (= y t1o13))
(and (= x t1o15) (= y t1o14))
(and (= x t1o15) (= y t1o16))
(and (= x t1o16) (= y t1o4))
(and (= x t1o16) (= y t1o8))
(and (= x t1o16) (= y t1o9))
(and (= x t1o16) (= y t1o13))
(and (= x t2o1) (= y t2o2))
(and (= x t2o1) (= y t2o6))
(and (= x t2o1) (= y t2o11))
(and (= x t2o3) (= y t2o2))
(and (= x t2o3) (= y t2o6))
(and (= x t2o3) (= y t2o11))
(and (= x t2o4) (= y t2o1))
(and (= x t2o4) (= y t2o2))
(and (= x t2o4) (= y t2o6))
(and (= x t2o4) (= y t2o7))
(and (= x t2o4) (= y t2o11))
(and (= x t2o4) (= y t2o15))
(and (= x t2o5) (= y t2o1))
(and (= x t2o5) (= y t2o2))
(and (= x t2o5) (= y t2o6))
(and (= x t2o5) (= y t2o7))
(and (= x t2o5) (= y t2o11))
(and (= x t2o5) (= y t2o15))
(and (= x t2o6) (= y t2o2))
(and (= x t2o7) (= y t2o2))
(and (= x t2o7) (= y t2o11))
(and (= x t2o8) (= y t2o1))
(and (= x t2o8) (= y t2o2))
(and (= x t2o8) (= y t2o5))
(and (= x t2o8) (= y t2o6))
(and (= x t2o8) (= y t2o7))
(and (= x t2o8) (= y t2o11))
(and (= x t2o8) (= y t2o12))
(and (= x t2o8) (= y t2o15))
(and (= x t2o9) (= y t2o1))
(and (= x t2o9) (= y t2o2))
(and (= x t2o9) (= y t2o4))
(and (= x t2o9) (= y t2o5))
(and (= x t2o9) (= y t2o6))
(and (= x t2o9) (= y t2o7))
(and (= x t2o9) (= y t2o11))
(and (= x t2o9) (= y t2o12))
(and (= x t2o9) (= y t2o15))
(and (= x t2o10) (= y t2o1))
(and (= x t2o10) (= y t2o2))
(and (= x t2o10) (= y t2o4))
(and (= x t2o10) (= y t2o5))
(and (= x t2o10) (= y t2o6))
(and (= x t2o10) (= y t2o7))
(and (= x t2o10) (= y t2o8))
(and (= x t2o10) (= y t2o9))
(and (= x t2o10) (= y t2o11))
(and (= x t2o10) (= y t2o12))
(and (= x t2o10) (= y t2o15))
(and (= x t2o11) (= y t2o2))
(and (= x t2o12) (= y t2o1))
(and (= x t2o12) (= y t2o2))
(and (= x t2o12) (= y t2o6))
(and (= x t2o12) (= y t2o7))
(and (= x t2o12) (= y t2o11))
(and (= x t2o12) (= y t2o15))
(and (= x t2o13) (= y t2o1))
(and (= x t2o13) (= y t2o2))
(and (= x t2o13) (= y t2o4))
(and (= x t2o13) (= y t2o5))
(and (= x t2o13) (= y t2o6))
(and (= x t2o13) (= y t2o7))
(and (= x t2o13) (= y t2o8))
(and (= x t2o13) (= y t2o9))
(and (= x t2o13) (= y t2o10))
(and (= x t2o13) (= y t2o11))
(and (= x t2o13) (= y t2o12))
(and (= x t2o13) (= y t2o15))
(and (= x t2o14) (= y t2o1))
(and (= x t2o14) (= y t2o2))
(and (= x t2o14) (= y t2o4))
(and (= x t2o14) (= y t2o5))
(and (= x t2o14) (= y t2o6))
(and (= x t2o14) (= y t2o7))
(and (= x t2o14) (= y t2o8))
(and (= x t2o14) (= y t2o9))
(and (= x t2o14) (= y t2o10))
(and (= x t2o14) (= y t2o11))
(and (= x t2o14) (= y t2o12))
(and (= x t2o14) (= y t2o13))
(and (= x t2o14) (= y t2o15))
(and (= x t2o15) (= y t2o2))
(and (= x t2o15) (= y t2o6))
(and (= x t2o15) (= y t2o11))
(and (= x c1o1) (= y c1o2))
(and (= x c1o1) (= y c1o3))
(and (= x c1o1) (= y c1o4))
(and (= x c1o4) (= y c1o2))
(and (= x c1o4) (= y c1o3))
(and (= x c1o5) (= y c1o2))
(and (= x c1o5) (= y c1o3))
(and (= x c2o2) (= y c2o1))
(and (= x c3o1) (= y c3o3))
(and (= x c3o2) (= y c3o3))
(and (= x c3o4) (= y c3o2))
(and (= x c3o4) (= y c3o3))

))

(define-fun above ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o3) (= y t1o1))
(and (= x t2o2) (= y t2o1))
(and (= x t2o2) (= y t2o4))
(and (= x t2o10) (= y t2o1))
(and (= x t2o12) (= y t2o1))
(and (= x t2o13) (= y t2o1))
(and (= x t2o13) (= y t2o2))
(and (= x t2o13) (= y t2o4))
(and (= x t2o13) (= y t2o5))
(and (= x t2o13) (= y t2o6))
(and (= x t2o13) (= y t2o7))
(and (= x t2o13) (= y t2o8))
(and (= x t2o13) (= y t2o9))
(and (= x t2o13) (= y t2o10))
(and (= x t2o13) (= y t2o11))
(and (= x t2o13) (= y t2o12))
(and (= x t2o13) (= y t2o15))
(and (= x t2o14) (= y t2o1))
(and (= x t2o14) (= y t2o2))
(and (= x t2o14) (= y t2o4))
(and (= x t2o14) (= y t2o5))
(and (= x t2o14) (= y t2o6))
(and (= x t2o14) (= y t2o7))
(and (= x t2o14) (= y t2o8))
(and (= x t2o14) (= y t2o9))
(and (= x t2o14) (= y t2o10))
(and (= x t2o14) (= y t2o11))
(and (= x t2o14) (= y t2o12))
(and (= x t2o14) (= y t2o15))
(and (= x c3o4) (= y c3o2))
(and (= x c3o4) (= y c3o3))

))

(define-fun below ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o3))
(and (= x t2o1) (= y t2o2))
(and (= x t2o1) (= y t2o10))
(and (= x t2o1) (= y t2o12))
(and (= x t2o1) (= y t2o13))
(and (= x t2o1) (= y t2o14))
(and (= x t2o2) (= y t2o13))
(and (= x t2o2) (= y t2o14))
(and (= x t2o4) (= y t2o2))
(and (= x t2o4) (= y t2o13))
(and (= x t2o4) (= y t2o14))
(and (= x t2o5) (= y t2o13))
(and (= x t2o5) (= y t2o14))
(and (= x t2o6) (= y t2o13))
(and (= x t2o6) (= y t2o14))
(and (= x t2o7) (= y t2o13))
(and (= x t2o7) (= y t2o14))
(and (= x t2o8) (= y t2o13))
(and (= x t2o8) (= y t2o14))
(and (= x t2o9) (= y t2o13))
(and (= x t2o9) (= y t2o14))
(and (= x t2o10) (= y t2o13))
(and (= x t2o10) (= y t2o14))
(and (= x t2o11) (= y t2o13))
(and (= x t2o11) (= y t2o14))
(and (= x t2o12) (= y t2o13))
(and (= x t2o12) (= y t2o14))
(and (= x t2o15) (= y t2o13))
(and (= x t2o15) (= y t2o14))
(and (= x c3o2) (= y c3o4))
(and (= x c3o3) (= y c3o4))

))

(define-fun within ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o7) (= y t1o2))
(and (= x t1o12) (= y t1o2))
(and (= x t2o1) (= y t2o3))
(and (= x t2o8) (= y t2o3))
(and (= x t2o9) (= y t2o3))
(and (= x t2o10) (= y t2o3))
(and (= x t2o12) (= y t2o3))
(and (= x t2o13) (= y t2o3))
(and (= x t2o14) (= y t2o3))
(and (= x t2o15) (= y t2o3))
(and (= x t3o2) (= y t3o1))
(and (= x c3o4) (= y c3o1))

))



(define-fun eq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (not (= x1 x2)))


(declare-const q1 Bool)
(declare-const q2 Bool)

(declare-const lbl_g1 (_ BitVec 3))
(assert (or
(= lbl_g1 person)
(= lbl_g1 car)
(= lbl_g1 bus)
(= lbl_g1 backpack)
(= lbl_g1 handbag)
)
)
(define-fun g1 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g1)
)

(declare-const lbl_g2 (_ BitVec 3))
(assert (or
(= lbl_g2 person)
(= lbl_g2 car)
(= lbl_g2 bus)
(= lbl_g2 backpack)
(= lbl_g2 handbag)
)
)
(define-fun g2 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g2)
)


(declare-const constval_person (_ BitVec 3))
(assert (= constval_person person))
(declare-const constval_car (_ BitVec 3))
(assert (= constval_car car))
(declare-const constval_bus (_ BitVec 3))
(assert (= constval_bus bus))
(declare-const constval_backpack (_ BitVec 3))
(assert (= constval_backpack backpack))
(declare-const constval_handbag (_ BitVec 3))
(assert (= constval_handbag handbag))

(declare-const eq_x1_x1 Bool)
(declare-const eq_x1_x2 Bool)
(declare-const eq_x2_x1 Bool)
(declare-const eq_x2_x2 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const neq_x1_x2 Bool)
(declare-const neq_x2_x1 Bool)
(declare-const neq_x2_x2 Bool)
(declare-const labelOf_x1_person Bool)
(declare-const labelOf_x1_car Bool)
(declare-const labelOf_x1_bus Bool)
(declare-const labelOf_x1_backpack Bool)
(declare-const labelOf_x1_handbag Bool)
(declare-const labelOf_x2_person Bool)
(declare-const labelOf_x2_car Bool)
(declare-const labelOf_x2_bus Bool)
(declare-const labelOf_x2_backpack Bool)
(declare-const labelOf_x2_handbag Bool)
(declare-const leftOf_x1_x1 Bool)
(declare-const leftOf_x1_x2 Bool)
(declare-const leftOf_x2_x1 Bool)
(declare-const leftOf_x2_x2 Bool)
(declare-const rightOf_x1_x1 Bool)
(declare-const rightOf_x1_x2 Bool)
(declare-const rightOf_x2_x1 Bool)
(declare-const rightOf_x2_x2 Bool)
(declare-const above_x1_x1 Bool)
(declare-const above_x1_x2 Bool)
(declare-const above_x2_x1 Bool)
(declare-const above_x2_x2 Bool)
(declare-const below_x1_x1 Bool)
(declare-const below_x1_x2 Bool)
(declare-const below_x2_x1 Bool)
(declare-const below_x2_x2 Bool)
(declare-const within_x1_x1 Bool)
(declare-const within_x1_x2 Bool)
(declare-const within_x2_x1 Bool)
(declare-const within_x2_x2 Bool)

(define-fun f ((x1 (_ BitVec 6)) (x2 (_ BitVec 6))) Bool
(and
(=>
eq_x1_x1
(eq x1 x1)
)
(=>
eq_x1_x2
(eq x1 x2)
)
(=>
eq_x2_x1
(eq x2 x1)
)
(=>
eq_x2_x2
(eq x2 x2)
)
(=>
neq_x1_x1
(neq x1 x1)
)
(=>
neq_x1_x2
(neq x1 x2)
)
(=>
neq_x2_x1
(neq x2 x1)
)
(=>
neq_x2_x2
(neq x2 x2)
)
(=>
labelOf_x1_person
(labelOf x1 person)
)
(=>
labelOf_x1_car
(labelOf x1 car)
)
(=>
labelOf_x1_bus
(labelOf x1 bus)
)
(=>
labelOf_x1_backpack
(labelOf x1 backpack)
)
(=>
labelOf_x1_handbag
(labelOf x1 handbag)
)
(=>
labelOf_x2_person
(labelOf x2 person)
)
(=>
labelOf_x2_car
(labelOf x2 car)
)
(=>
labelOf_x2_bus
(labelOf x2 bus)
)
(=>
labelOf_x2_backpack
(labelOf x2 backpack)
)
(=>
labelOf_x2_handbag
(labelOf x2 handbag)
)
(=>
leftOf_x1_x1
(leftOf x1 x1)
)
(=>
leftOf_x1_x2
(leftOf x1 x2)
)
(=>
leftOf_x2_x1
(leftOf x2 x1)
)
(=>
leftOf_x2_x2
(leftOf x2 x2)
)
(=>
rightOf_x1_x1
(rightOf x1 x1)
)
(=>
rightOf_x1_x2
(rightOf x1 x2)
)
(=>
rightOf_x2_x1
(rightOf x2 x1)
)
(=>
rightOf_x2_x2
(rightOf x2 x2)
)
(=>
above_x1_x1
(above x1 x1)
)
(=>
above_x1_x2
(above x1 x2)
)
(=>
above_x2_x1
(above x2 x1)
)
(=>
above_x2_x2
(above x2 x2)
)
(=>
below_x1_x1
(below x1 x1)
)
(=>
below_x1_x2
(below x1 x2)
)
(=>
below_x2_x1
(below x2 x1)
)
(=>
below_x2_x2
(below x2 x2)
)
(=>
within_x1_x1
(within x1 x1)
)
(=>
within_x1_x2
(within x1 x2)
)
(=>
within_x2_x1
(within x2 x1)
)
(=>
within_x2_x2
(within x2 x2)
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
(= q2 true)
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
(g1 t1o9)
(g1 t1o10)
(g1 t1o11)
(g1 t1o12)
(g1 t1o13)
(g1 t1o14)
(g1 t1o15)
(g1 t1o16)
)
(and
(=>
(g1 t1o1)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o1 t1o1)
)
(=>
(g2 t1o2)
(f t1o1 t1o2)
)
(=>
(g2 t1o3)
(f t1o1 t1o3)
)
(=>
(g2 t1o4)
(f t1o1 t1o4)
)
(=>
(g2 t1o5)
(f t1o1 t1o5)
)
(=>
(g2 t1o6)
(f t1o1 t1o6)
)
(=>
(g2 t1o7)
(f t1o1 t1o7)
)
(=>
(g2 t1o8)
(f t1o1 t1o8)
)
(=>
(g2 t1o9)
(f t1o1 t1o9)
)
(=>
(g2 t1o10)
(f t1o1 t1o10)
)
(=>
(g2 t1o11)
(f t1o1 t1o11)
)
(=>
(g2 t1o12)
(f t1o1 t1o12)
)
(=>
(g2 t1o13)
(f t1o1 t1o13)
)
(=>
(g2 t1o14)
(f t1o1 t1o14)
)
(=>
(g2 t1o15)
(f t1o1 t1o15)
)
(=>
(g2 t1o16)
(f t1o1 t1o16)
)
)
)
)
(=>
(g1 t1o2)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o2 t1o1)
)
(=>
(g2 t1o2)
(f t1o2 t1o2)
)
(=>
(g2 t1o3)
(f t1o2 t1o3)
)
(=>
(g2 t1o4)
(f t1o2 t1o4)
)
(=>
(g2 t1o5)
(f t1o2 t1o5)
)
(=>
(g2 t1o6)
(f t1o2 t1o6)
)
(=>
(g2 t1o7)
(f t1o2 t1o7)
)
(=>
(g2 t1o8)
(f t1o2 t1o8)
)
(=>
(g2 t1o9)
(f t1o2 t1o9)
)
(=>
(g2 t1o10)
(f t1o2 t1o10)
)
(=>
(g2 t1o11)
(f t1o2 t1o11)
)
(=>
(g2 t1o12)
(f t1o2 t1o12)
)
(=>
(g2 t1o13)
(f t1o2 t1o13)
)
(=>
(g2 t1o14)
(f t1o2 t1o14)
)
(=>
(g2 t1o15)
(f t1o2 t1o15)
)
(=>
(g2 t1o16)
(f t1o2 t1o16)
)
)
)
)
(=>
(g1 t1o3)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o3 t1o1)
)
(=>
(g2 t1o2)
(f t1o3 t1o2)
)
(=>
(g2 t1o3)
(f t1o3 t1o3)
)
(=>
(g2 t1o4)
(f t1o3 t1o4)
)
(=>
(g2 t1o5)
(f t1o3 t1o5)
)
(=>
(g2 t1o6)
(f t1o3 t1o6)
)
(=>
(g2 t1o7)
(f t1o3 t1o7)
)
(=>
(g2 t1o8)
(f t1o3 t1o8)
)
(=>
(g2 t1o9)
(f t1o3 t1o9)
)
(=>
(g2 t1o10)
(f t1o3 t1o10)
)
(=>
(g2 t1o11)
(f t1o3 t1o11)
)
(=>
(g2 t1o12)
(f t1o3 t1o12)
)
(=>
(g2 t1o13)
(f t1o3 t1o13)
)
(=>
(g2 t1o14)
(f t1o3 t1o14)
)
(=>
(g2 t1o15)
(f t1o3 t1o15)
)
(=>
(g2 t1o16)
(f t1o3 t1o16)
)
)
)
)
(=>
(g1 t1o4)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o4 t1o1)
)
(=>
(g2 t1o2)
(f t1o4 t1o2)
)
(=>
(g2 t1o3)
(f t1o4 t1o3)
)
(=>
(g2 t1o4)
(f t1o4 t1o4)
)
(=>
(g2 t1o5)
(f t1o4 t1o5)
)
(=>
(g2 t1o6)
(f t1o4 t1o6)
)
(=>
(g2 t1o7)
(f t1o4 t1o7)
)
(=>
(g2 t1o8)
(f t1o4 t1o8)
)
(=>
(g2 t1o9)
(f t1o4 t1o9)
)
(=>
(g2 t1o10)
(f t1o4 t1o10)
)
(=>
(g2 t1o11)
(f t1o4 t1o11)
)
(=>
(g2 t1o12)
(f t1o4 t1o12)
)
(=>
(g2 t1o13)
(f t1o4 t1o13)
)
(=>
(g2 t1o14)
(f t1o4 t1o14)
)
(=>
(g2 t1o15)
(f t1o4 t1o15)
)
(=>
(g2 t1o16)
(f t1o4 t1o16)
)
)
)
)
(=>
(g1 t1o5)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o5 t1o1)
)
(=>
(g2 t1o2)
(f t1o5 t1o2)
)
(=>
(g2 t1o3)
(f t1o5 t1o3)
)
(=>
(g2 t1o4)
(f t1o5 t1o4)
)
(=>
(g2 t1o5)
(f t1o5 t1o5)
)
(=>
(g2 t1o6)
(f t1o5 t1o6)
)
(=>
(g2 t1o7)
(f t1o5 t1o7)
)
(=>
(g2 t1o8)
(f t1o5 t1o8)
)
(=>
(g2 t1o9)
(f t1o5 t1o9)
)
(=>
(g2 t1o10)
(f t1o5 t1o10)
)
(=>
(g2 t1o11)
(f t1o5 t1o11)
)
(=>
(g2 t1o12)
(f t1o5 t1o12)
)
(=>
(g2 t1o13)
(f t1o5 t1o13)
)
(=>
(g2 t1o14)
(f t1o5 t1o14)
)
(=>
(g2 t1o15)
(f t1o5 t1o15)
)
(=>
(g2 t1o16)
(f t1o5 t1o16)
)
)
)
)
(=>
(g1 t1o6)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o6 t1o1)
)
(=>
(g2 t1o2)
(f t1o6 t1o2)
)
(=>
(g2 t1o3)
(f t1o6 t1o3)
)
(=>
(g2 t1o4)
(f t1o6 t1o4)
)
(=>
(g2 t1o5)
(f t1o6 t1o5)
)
(=>
(g2 t1o6)
(f t1o6 t1o6)
)
(=>
(g2 t1o7)
(f t1o6 t1o7)
)
(=>
(g2 t1o8)
(f t1o6 t1o8)
)
(=>
(g2 t1o9)
(f t1o6 t1o9)
)
(=>
(g2 t1o10)
(f t1o6 t1o10)
)
(=>
(g2 t1o11)
(f t1o6 t1o11)
)
(=>
(g2 t1o12)
(f t1o6 t1o12)
)
(=>
(g2 t1o13)
(f t1o6 t1o13)
)
(=>
(g2 t1o14)
(f t1o6 t1o14)
)
(=>
(g2 t1o15)
(f t1o6 t1o15)
)
(=>
(g2 t1o16)
(f t1o6 t1o16)
)
)
)
)
(=>
(g1 t1o7)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o7 t1o1)
)
(=>
(g2 t1o2)
(f t1o7 t1o2)
)
(=>
(g2 t1o3)
(f t1o7 t1o3)
)
(=>
(g2 t1o4)
(f t1o7 t1o4)
)
(=>
(g2 t1o5)
(f t1o7 t1o5)
)
(=>
(g2 t1o6)
(f t1o7 t1o6)
)
(=>
(g2 t1o7)
(f t1o7 t1o7)
)
(=>
(g2 t1o8)
(f t1o7 t1o8)
)
(=>
(g2 t1o9)
(f t1o7 t1o9)
)
(=>
(g2 t1o10)
(f t1o7 t1o10)
)
(=>
(g2 t1o11)
(f t1o7 t1o11)
)
(=>
(g2 t1o12)
(f t1o7 t1o12)
)
(=>
(g2 t1o13)
(f t1o7 t1o13)
)
(=>
(g2 t1o14)
(f t1o7 t1o14)
)
(=>
(g2 t1o15)
(f t1o7 t1o15)
)
(=>
(g2 t1o16)
(f t1o7 t1o16)
)
)
)
)
(=>
(g1 t1o8)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o8 t1o1)
)
(=>
(g2 t1o2)
(f t1o8 t1o2)
)
(=>
(g2 t1o3)
(f t1o8 t1o3)
)
(=>
(g2 t1o4)
(f t1o8 t1o4)
)
(=>
(g2 t1o5)
(f t1o8 t1o5)
)
(=>
(g2 t1o6)
(f t1o8 t1o6)
)
(=>
(g2 t1o7)
(f t1o8 t1o7)
)
(=>
(g2 t1o8)
(f t1o8 t1o8)
)
(=>
(g2 t1o9)
(f t1o8 t1o9)
)
(=>
(g2 t1o10)
(f t1o8 t1o10)
)
(=>
(g2 t1o11)
(f t1o8 t1o11)
)
(=>
(g2 t1o12)
(f t1o8 t1o12)
)
(=>
(g2 t1o13)
(f t1o8 t1o13)
)
(=>
(g2 t1o14)
(f t1o8 t1o14)
)
(=>
(g2 t1o15)
(f t1o8 t1o15)
)
(=>
(g2 t1o16)
(f t1o8 t1o16)
)
)
)
)
(=>
(g1 t1o9)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o9 t1o1)
)
(=>
(g2 t1o2)
(f t1o9 t1o2)
)
(=>
(g2 t1o3)
(f t1o9 t1o3)
)
(=>
(g2 t1o4)
(f t1o9 t1o4)
)
(=>
(g2 t1o5)
(f t1o9 t1o5)
)
(=>
(g2 t1o6)
(f t1o9 t1o6)
)
(=>
(g2 t1o7)
(f t1o9 t1o7)
)
(=>
(g2 t1o8)
(f t1o9 t1o8)
)
(=>
(g2 t1o9)
(f t1o9 t1o9)
)
(=>
(g2 t1o10)
(f t1o9 t1o10)
)
(=>
(g2 t1o11)
(f t1o9 t1o11)
)
(=>
(g2 t1o12)
(f t1o9 t1o12)
)
(=>
(g2 t1o13)
(f t1o9 t1o13)
)
(=>
(g2 t1o14)
(f t1o9 t1o14)
)
(=>
(g2 t1o15)
(f t1o9 t1o15)
)
(=>
(g2 t1o16)
(f t1o9 t1o16)
)
)
)
)
(=>
(g1 t1o10)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o10 t1o1)
)
(=>
(g2 t1o2)
(f t1o10 t1o2)
)
(=>
(g2 t1o3)
(f t1o10 t1o3)
)
(=>
(g2 t1o4)
(f t1o10 t1o4)
)
(=>
(g2 t1o5)
(f t1o10 t1o5)
)
(=>
(g2 t1o6)
(f t1o10 t1o6)
)
(=>
(g2 t1o7)
(f t1o10 t1o7)
)
(=>
(g2 t1o8)
(f t1o10 t1o8)
)
(=>
(g2 t1o9)
(f t1o10 t1o9)
)
(=>
(g2 t1o10)
(f t1o10 t1o10)
)
(=>
(g2 t1o11)
(f t1o10 t1o11)
)
(=>
(g2 t1o12)
(f t1o10 t1o12)
)
(=>
(g2 t1o13)
(f t1o10 t1o13)
)
(=>
(g2 t1o14)
(f t1o10 t1o14)
)
(=>
(g2 t1o15)
(f t1o10 t1o15)
)
(=>
(g2 t1o16)
(f t1o10 t1o16)
)
)
)
)
(=>
(g1 t1o11)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o11 t1o1)
)
(=>
(g2 t1o2)
(f t1o11 t1o2)
)
(=>
(g2 t1o3)
(f t1o11 t1o3)
)
(=>
(g2 t1o4)
(f t1o11 t1o4)
)
(=>
(g2 t1o5)
(f t1o11 t1o5)
)
(=>
(g2 t1o6)
(f t1o11 t1o6)
)
(=>
(g2 t1o7)
(f t1o11 t1o7)
)
(=>
(g2 t1o8)
(f t1o11 t1o8)
)
(=>
(g2 t1o9)
(f t1o11 t1o9)
)
(=>
(g2 t1o10)
(f t1o11 t1o10)
)
(=>
(g2 t1o11)
(f t1o11 t1o11)
)
(=>
(g2 t1o12)
(f t1o11 t1o12)
)
(=>
(g2 t1o13)
(f t1o11 t1o13)
)
(=>
(g2 t1o14)
(f t1o11 t1o14)
)
(=>
(g2 t1o15)
(f t1o11 t1o15)
)
(=>
(g2 t1o16)
(f t1o11 t1o16)
)
)
)
)
(=>
(g1 t1o12)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o12 t1o1)
)
(=>
(g2 t1o2)
(f t1o12 t1o2)
)
(=>
(g2 t1o3)
(f t1o12 t1o3)
)
(=>
(g2 t1o4)
(f t1o12 t1o4)
)
(=>
(g2 t1o5)
(f t1o12 t1o5)
)
(=>
(g2 t1o6)
(f t1o12 t1o6)
)
(=>
(g2 t1o7)
(f t1o12 t1o7)
)
(=>
(g2 t1o8)
(f t1o12 t1o8)
)
(=>
(g2 t1o9)
(f t1o12 t1o9)
)
(=>
(g2 t1o10)
(f t1o12 t1o10)
)
(=>
(g2 t1o11)
(f t1o12 t1o11)
)
(=>
(g2 t1o12)
(f t1o12 t1o12)
)
(=>
(g2 t1o13)
(f t1o12 t1o13)
)
(=>
(g2 t1o14)
(f t1o12 t1o14)
)
(=>
(g2 t1o15)
(f t1o12 t1o15)
)
(=>
(g2 t1o16)
(f t1o12 t1o16)
)
)
)
)
(=>
(g1 t1o13)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o13 t1o1)
)
(=>
(g2 t1o2)
(f t1o13 t1o2)
)
(=>
(g2 t1o3)
(f t1o13 t1o3)
)
(=>
(g2 t1o4)
(f t1o13 t1o4)
)
(=>
(g2 t1o5)
(f t1o13 t1o5)
)
(=>
(g2 t1o6)
(f t1o13 t1o6)
)
(=>
(g2 t1o7)
(f t1o13 t1o7)
)
(=>
(g2 t1o8)
(f t1o13 t1o8)
)
(=>
(g2 t1o9)
(f t1o13 t1o9)
)
(=>
(g2 t1o10)
(f t1o13 t1o10)
)
(=>
(g2 t1o11)
(f t1o13 t1o11)
)
(=>
(g2 t1o12)
(f t1o13 t1o12)
)
(=>
(g2 t1o13)
(f t1o13 t1o13)
)
(=>
(g2 t1o14)
(f t1o13 t1o14)
)
(=>
(g2 t1o15)
(f t1o13 t1o15)
)
(=>
(g2 t1o16)
(f t1o13 t1o16)
)
)
)
)
(=>
(g1 t1o14)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o14 t1o1)
)
(=>
(g2 t1o2)
(f t1o14 t1o2)
)
(=>
(g2 t1o3)
(f t1o14 t1o3)
)
(=>
(g2 t1o4)
(f t1o14 t1o4)
)
(=>
(g2 t1o5)
(f t1o14 t1o5)
)
(=>
(g2 t1o6)
(f t1o14 t1o6)
)
(=>
(g2 t1o7)
(f t1o14 t1o7)
)
(=>
(g2 t1o8)
(f t1o14 t1o8)
)
(=>
(g2 t1o9)
(f t1o14 t1o9)
)
(=>
(g2 t1o10)
(f t1o14 t1o10)
)
(=>
(g2 t1o11)
(f t1o14 t1o11)
)
(=>
(g2 t1o12)
(f t1o14 t1o12)
)
(=>
(g2 t1o13)
(f t1o14 t1o13)
)
(=>
(g2 t1o14)
(f t1o14 t1o14)
)
(=>
(g2 t1o15)
(f t1o14 t1o15)
)
(=>
(g2 t1o16)
(f t1o14 t1o16)
)
)
)
)
(=>
(g1 t1o15)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o15 t1o1)
)
(=>
(g2 t1o2)
(f t1o15 t1o2)
)
(=>
(g2 t1o3)
(f t1o15 t1o3)
)
(=>
(g2 t1o4)
(f t1o15 t1o4)
)
(=>
(g2 t1o5)
(f t1o15 t1o5)
)
(=>
(g2 t1o6)
(f t1o15 t1o6)
)
(=>
(g2 t1o7)
(f t1o15 t1o7)
)
(=>
(g2 t1o8)
(f t1o15 t1o8)
)
(=>
(g2 t1o9)
(f t1o15 t1o9)
)
(=>
(g2 t1o10)
(f t1o15 t1o10)
)
(=>
(g2 t1o11)
(f t1o15 t1o11)
)
(=>
(g2 t1o12)
(f t1o15 t1o12)
)
(=>
(g2 t1o13)
(f t1o15 t1o13)
)
(=>
(g2 t1o14)
(f t1o15 t1o14)
)
(=>
(g2 t1o15)
(f t1o15 t1o15)
)
(=>
(g2 t1o16)
(f t1o15 t1o16)
)
)
)
)
(=>
(g1 t1o16)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o16 t1o1)
)
(=>
(g2 t1o2)
(f t1o16 t1o2)
)
(=>
(g2 t1o3)
(f t1o16 t1o3)
)
(=>
(g2 t1o4)
(f t1o16 t1o4)
)
(=>
(g2 t1o5)
(f t1o16 t1o5)
)
(=>
(g2 t1o6)
(f t1o16 t1o6)
)
(=>
(g2 t1o7)
(f t1o16 t1o7)
)
(=>
(g2 t1o8)
(f t1o16 t1o8)
)
(=>
(g2 t1o9)
(f t1o16 t1o9)
)
(=>
(g2 t1o10)
(f t1o16 t1o10)
)
(=>
(g2 t1o11)
(f t1o16 t1o11)
)
(=>
(g2 t1o12)
(f t1o16 t1o12)
)
(=>
(g2 t1o13)
(f t1o16 t1o13)
)
(=>
(g2 t1o14)
(f t1o16 t1o14)
)
(=>
(g2 t1o15)
(f t1o16 t1o15)
)
(=>
(g2 t1o16)
(f t1o16 t1o16)
)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
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
(g1 t1o9)
(g1 t1o10)
(g1 t1o11)
(g1 t1o12)
(g1 t1o13)
(g1 t1o14)
(g1 t1o15)
(g1 t1o16)
)
(and
(=>
(g1 t1o1)
(or
(and
(g2 t1o1)
(f t1o1 t1o1)
)
(and
(g2 t1o2)
(f t1o1 t1o2)
)
(and
(g2 t1o3)
(f t1o1 t1o3)
)
(and
(g2 t1o4)
(f t1o1 t1o4)
)
(and
(g2 t1o5)
(f t1o1 t1o5)
)
(and
(g2 t1o6)
(f t1o1 t1o6)
)
(and
(g2 t1o7)
(f t1o1 t1o7)
)
(and
(g2 t1o8)
(f t1o1 t1o8)
)
(and
(g2 t1o9)
(f t1o1 t1o9)
)
(and
(g2 t1o10)
(f t1o1 t1o10)
)
(and
(g2 t1o11)
(f t1o1 t1o11)
)
(and
(g2 t1o12)
(f t1o1 t1o12)
)
(and
(g2 t1o13)
(f t1o1 t1o13)
)
(and
(g2 t1o14)
(f t1o1 t1o14)
)
(and
(g2 t1o15)
(f t1o1 t1o15)
)
(and
(g2 t1o16)
(f t1o1 t1o16)
)
)
)
(=>
(g1 t1o2)
(or
(and
(g2 t1o1)
(f t1o2 t1o1)
)
(and
(g2 t1o2)
(f t1o2 t1o2)
)
(and
(g2 t1o3)
(f t1o2 t1o3)
)
(and
(g2 t1o4)
(f t1o2 t1o4)
)
(and
(g2 t1o5)
(f t1o2 t1o5)
)
(and
(g2 t1o6)
(f t1o2 t1o6)
)
(and
(g2 t1o7)
(f t1o2 t1o7)
)
(and
(g2 t1o8)
(f t1o2 t1o8)
)
(and
(g2 t1o9)
(f t1o2 t1o9)
)
(and
(g2 t1o10)
(f t1o2 t1o10)
)
(and
(g2 t1o11)
(f t1o2 t1o11)
)
(and
(g2 t1o12)
(f t1o2 t1o12)
)
(and
(g2 t1o13)
(f t1o2 t1o13)
)
(and
(g2 t1o14)
(f t1o2 t1o14)
)
(and
(g2 t1o15)
(f t1o2 t1o15)
)
(and
(g2 t1o16)
(f t1o2 t1o16)
)
)
)
(=>
(g1 t1o3)
(or
(and
(g2 t1o1)
(f t1o3 t1o1)
)
(and
(g2 t1o2)
(f t1o3 t1o2)
)
(and
(g2 t1o3)
(f t1o3 t1o3)
)
(and
(g2 t1o4)
(f t1o3 t1o4)
)
(and
(g2 t1o5)
(f t1o3 t1o5)
)
(and
(g2 t1o6)
(f t1o3 t1o6)
)
(and
(g2 t1o7)
(f t1o3 t1o7)
)
(and
(g2 t1o8)
(f t1o3 t1o8)
)
(and
(g2 t1o9)
(f t1o3 t1o9)
)
(and
(g2 t1o10)
(f t1o3 t1o10)
)
(and
(g2 t1o11)
(f t1o3 t1o11)
)
(and
(g2 t1o12)
(f t1o3 t1o12)
)
(and
(g2 t1o13)
(f t1o3 t1o13)
)
(and
(g2 t1o14)
(f t1o3 t1o14)
)
(and
(g2 t1o15)
(f t1o3 t1o15)
)
(and
(g2 t1o16)
(f t1o3 t1o16)
)
)
)
(=>
(g1 t1o4)
(or
(and
(g2 t1o1)
(f t1o4 t1o1)
)
(and
(g2 t1o2)
(f t1o4 t1o2)
)
(and
(g2 t1o3)
(f t1o4 t1o3)
)
(and
(g2 t1o4)
(f t1o4 t1o4)
)
(and
(g2 t1o5)
(f t1o4 t1o5)
)
(and
(g2 t1o6)
(f t1o4 t1o6)
)
(and
(g2 t1o7)
(f t1o4 t1o7)
)
(and
(g2 t1o8)
(f t1o4 t1o8)
)
(and
(g2 t1o9)
(f t1o4 t1o9)
)
(and
(g2 t1o10)
(f t1o4 t1o10)
)
(and
(g2 t1o11)
(f t1o4 t1o11)
)
(and
(g2 t1o12)
(f t1o4 t1o12)
)
(and
(g2 t1o13)
(f t1o4 t1o13)
)
(and
(g2 t1o14)
(f t1o4 t1o14)
)
(and
(g2 t1o15)
(f t1o4 t1o15)
)
(and
(g2 t1o16)
(f t1o4 t1o16)
)
)
)
(=>
(g1 t1o5)
(or
(and
(g2 t1o1)
(f t1o5 t1o1)
)
(and
(g2 t1o2)
(f t1o5 t1o2)
)
(and
(g2 t1o3)
(f t1o5 t1o3)
)
(and
(g2 t1o4)
(f t1o5 t1o4)
)
(and
(g2 t1o5)
(f t1o5 t1o5)
)
(and
(g2 t1o6)
(f t1o5 t1o6)
)
(and
(g2 t1o7)
(f t1o5 t1o7)
)
(and
(g2 t1o8)
(f t1o5 t1o8)
)
(and
(g2 t1o9)
(f t1o5 t1o9)
)
(and
(g2 t1o10)
(f t1o5 t1o10)
)
(and
(g2 t1o11)
(f t1o5 t1o11)
)
(and
(g2 t1o12)
(f t1o5 t1o12)
)
(and
(g2 t1o13)
(f t1o5 t1o13)
)
(and
(g2 t1o14)
(f t1o5 t1o14)
)
(and
(g2 t1o15)
(f t1o5 t1o15)
)
(and
(g2 t1o16)
(f t1o5 t1o16)
)
)
)
(=>
(g1 t1o6)
(or
(and
(g2 t1o1)
(f t1o6 t1o1)
)
(and
(g2 t1o2)
(f t1o6 t1o2)
)
(and
(g2 t1o3)
(f t1o6 t1o3)
)
(and
(g2 t1o4)
(f t1o6 t1o4)
)
(and
(g2 t1o5)
(f t1o6 t1o5)
)
(and
(g2 t1o6)
(f t1o6 t1o6)
)
(and
(g2 t1o7)
(f t1o6 t1o7)
)
(and
(g2 t1o8)
(f t1o6 t1o8)
)
(and
(g2 t1o9)
(f t1o6 t1o9)
)
(and
(g2 t1o10)
(f t1o6 t1o10)
)
(and
(g2 t1o11)
(f t1o6 t1o11)
)
(and
(g2 t1o12)
(f t1o6 t1o12)
)
(and
(g2 t1o13)
(f t1o6 t1o13)
)
(and
(g2 t1o14)
(f t1o6 t1o14)
)
(and
(g2 t1o15)
(f t1o6 t1o15)
)
(and
(g2 t1o16)
(f t1o6 t1o16)
)
)
)
(=>
(g1 t1o7)
(or
(and
(g2 t1o1)
(f t1o7 t1o1)
)
(and
(g2 t1o2)
(f t1o7 t1o2)
)
(and
(g2 t1o3)
(f t1o7 t1o3)
)
(and
(g2 t1o4)
(f t1o7 t1o4)
)
(and
(g2 t1o5)
(f t1o7 t1o5)
)
(and
(g2 t1o6)
(f t1o7 t1o6)
)
(and
(g2 t1o7)
(f t1o7 t1o7)
)
(and
(g2 t1o8)
(f t1o7 t1o8)
)
(and
(g2 t1o9)
(f t1o7 t1o9)
)
(and
(g2 t1o10)
(f t1o7 t1o10)
)
(and
(g2 t1o11)
(f t1o7 t1o11)
)
(and
(g2 t1o12)
(f t1o7 t1o12)
)
(and
(g2 t1o13)
(f t1o7 t1o13)
)
(and
(g2 t1o14)
(f t1o7 t1o14)
)
(and
(g2 t1o15)
(f t1o7 t1o15)
)
(and
(g2 t1o16)
(f t1o7 t1o16)
)
)
)
(=>
(g1 t1o8)
(or
(and
(g2 t1o1)
(f t1o8 t1o1)
)
(and
(g2 t1o2)
(f t1o8 t1o2)
)
(and
(g2 t1o3)
(f t1o8 t1o3)
)
(and
(g2 t1o4)
(f t1o8 t1o4)
)
(and
(g2 t1o5)
(f t1o8 t1o5)
)
(and
(g2 t1o6)
(f t1o8 t1o6)
)
(and
(g2 t1o7)
(f t1o8 t1o7)
)
(and
(g2 t1o8)
(f t1o8 t1o8)
)
(and
(g2 t1o9)
(f t1o8 t1o9)
)
(and
(g2 t1o10)
(f t1o8 t1o10)
)
(and
(g2 t1o11)
(f t1o8 t1o11)
)
(and
(g2 t1o12)
(f t1o8 t1o12)
)
(and
(g2 t1o13)
(f t1o8 t1o13)
)
(and
(g2 t1o14)
(f t1o8 t1o14)
)
(and
(g2 t1o15)
(f t1o8 t1o15)
)
(and
(g2 t1o16)
(f t1o8 t1o16)
)
)
)
(=>
(g1 t1o9)
(or
(and
(g2 t1o1)
(f t1o9 t1o1)
)
(and
(g2 t1o2)
(f t1o9 t1o2)
)
(and
(g2 t1o3)
(f t1o9 t1o3)
)
(and
(g2 t1o4)
(f t1o9 t1o4)
)
(and
(g2 t1o5)
(f t1o9 t1o5)
)
(and
(g2 t1o6)
(f t1o9 t1o6)
)
(and
(g2 t1o7)
(f t1o9 t1o7)
)
(and
(g2 t1o8)
(f t1o9 t1o8)
)
(and
(g2 t1o9)
(f t1o9 t1o9)
)
(and
(g2 t1o10)
(f t1o9 t1o10)
)
(and
(g2 t1o11)
(f t1o9 t1o11)
)
(and
(g2 t1o12)
(f t1o9 t1o12)
)
(and
(g2 t1o13)
(f t1o9 t1o13)
)
(and
(g2 t1o14)
(f t1o9 t1o14)
)
(and
(g2 t1o15)
(f t1o9 t1o15)
)
(and
(g2 t1o16)
(f t1o9 t1o16)
)
)
)
(=>
(g1 t1o10)
(or
(and
(g2 t1o1)
(f t1o10 t1o1)
)
(and
(g2 t1o2)
(f t1o10 t1o2)
)
(and
(g2 t1o3)
(f t1o10 t1o3)
)
(and
(g2 t1o4)
(f t1o10 t1o4)
)
(and
(g2 t1o5)
(f t1o10 t1o5)
)
(and
(g2 t1o6)
(f t1o10 t1o6)
)
(and
(g2 t1o7)
(f t1o10 t1o7)
)
(and
(g2 t1o8)
(f t1o10 t1o8)
)
(and
(g2 t1o9)
(f t1o10 t1o9)
)
(and
(g2 t1o10)
(f t1o10 t1o10)
)
(and
(g2 t1o11)
(f t1o10 t1o11)
)
(and
(g2 t1o12)
(f t1o10 t1o12)
)
(and
(g2 t1o13)
(f t1o10 t1o13)
)
(and
(g2 t1o14)
(f t1o10 t1o14)
)
(and
(g2 t1o15)
(f t1o10 t1o15)
)
(and
(g2 t1o16)
(f t1o10 t1o16)
)
)
)
(=>
(g1 t1o11)
(or
(and
(g2 t1o1)
(f t1o11 t1o1)
)
(and
(g2 t1o2)
(f t1o11 t1o2)
)
(and
(g2 t1o3)
(f t1o11 t1o3)
)
(and
(g2 t1o4)
(f t1o11 t1o4)
)
(and
(g2 t1o5)
(f t1o11 t1o5)
)
(and
(g2 t1o6)
(f t1o11 t1o6)
)
(and
(g2 t1o7)
(f t1o11 t1o7)
)
(and
(g2 t1o8)
(f t1o11 t1o8)
)
(and
(g2 t1o9)
(f t1o11 t1o9)
)
(and
(g2 t1o10)
(f t1o11 t1o10)
)
(and
(g2 t1o11)
(f t1o11 t1o11)
)
(and
(g2 t1o12)
(f t1o11 t1o12)
)
(and
(g2 t1o13)
(f t1o11 t1o13)
)
(and
(g2 t1o14)
(f t1o11 t1o14)
)
(and
(g2 t1o15)
(f t1o11 t1o15)
)
(and
(g2 t1o16)
(f t1o11 t1o16)
)
)
)
(=>
(g1 t1o12)
(or
(and
(g2 t1o1)
(f t1o12 t1o1)
)
(and
(g2 t1o2)
(f t1o12 t1o2)
)
(and
(g2 t1o3)
(f t1o12 t1o3)
)
(and
(g2 t1o4)
(f t1o12 t1o4)
)
(and
(g2 t1o5)
(f t1o12 t1o5)
)
(and
(g2 t1o6)
(f t1o12 t1o6)
)
(and
(g2 t1o7)
(f t1o12 t1o7)
)
(and
(g2 t1o8)
(f t1o12 t1o8)
)
(and
(g2 t1o9)
(f t1o12 t1o9)
)
(and
(g2 t1o10)
(f t1o12 t1o10)
)
(and
(g2 t1o11)
(f t1o12 t1o11)
)
(and
(g2 t1o12)
(f t1o12 t1o12)
)
(and
(g2 t1o13)
(f t1o12 t1o13)
)
(and
(g2 t1o14)
(f t1o12 t1o14)
)
(and
(g2 t1o15)
(f t1o12 t1o15)
)
(and
(g2 t1o16)
(f t1o12 t1o16)
)
)
)
(=>
(g1 t1o13)
(or
(and
(g2 t1o1)
(f t1o13 t1o1)
)
(and
(g2 t1o2)
(f t1o13 t1o2)
)
(and
(g2 t1o3)
(f t1o13 t1o3)
)
(and
(g2 t1o4)
(f t1o13 t1o4)
)
(and
(g2 t1o5)
(f t1o13 t1o5)
)
(and
(g2 t1o6)
(f t1o13 t1o6)
)
(and
(g2 t1o7)
(f t1o13 t1o7)
)
(and
(g2 t1o8)
(f t1o13 t1o8)
)
(and
(g2 t1o9)
(f t1o13 t1o9)
)
(and
(g2 t1o10)
(f t1o13 t1o10)
)
(and
(g2 t1o11)
(f t1o13 t1o11)
)
(and
(g2 t1o12)
(f t1o13 t1o12)
)
(and
(g2 t1o13)
(f t1o13 t1o13)
)
(and
(g2 t1o14)
(f t1o13 t1o14)
)
(and
(g2 t1o15)
(f t1o13 t1o15)
)
(and
(g2 t1o16)
(f t1o13 t1o16)
)
)
)
(=>
(g1 t1o14)
(or
(and
(g2 t1o1)
(f t1o14 t1o1)
)
(and
(g2 t1o2)
(f t1o14 t1o2)
)
(and
(g2 t1o3)
(f t1o14 t1o3)
)
(and
(g2 t1o4)
(f t1o14 t1o4)
)
(and
(g2 t1o5)
(f t1o14 t1o5)
)
(and
(g2 t1o6)
(f t1o14 t1o6)
)
(and
(g2 t1o7)
(f t1o14 t1o7)
)
(and
(g2 t1o8)
(f t1o14 t1o8)
)
(and
(g2 t1o9)
(f t1o14 t1o9)
)
(and
(g2 t1o10)
(f t1o14 t1o10)
)
(and
(g2 t1o11)
(f t1o14 t1o11)
)
(and
(g2 t1o12)
(f t1o14 t1o12)
)
(and
(g2 t1o13)
(f t1o14 t1o13)
)
(and
(g2 t1o14)
(f t1o14 t1o14)
)
(and
(g2 t1o15)
(f t1o14 t1o15)
)
(and
(g2 t1o16)
(f t1o14 t1o16)
)
)
)
(=>
(g1 t1o15)
(or
(and
(g2 t1o1)
(f t1o15 t1o1)
)
(and
(g2 t1o2)
(f t1o15 t1o2)
)
(and
(g2 t1o3)
(f t1o15 t1o3)
)
(and
(g2 t1o4)
(f t1o15 t1o4)
)
(and
(g2 t1o5)
(f t1o15 t1o5)
)
(and
(g2 t1o6)
(f t1o15 t1o6)
)
(and
(g2 t1o7)
(f t1o15 t1o7)
)
(and
(g2 t1o8)
(f t1o15 t1o8)
)
(and
(g2 t1o9)
(f t1o15 t1o9)
)
(and
(g2 t1o10)
(f t1o15 t1o10)
)
(and
(g2 t1o11)
(f t1o15 t1o11)
)
(and
(g2 t1o12)
(f t1o15 t1o12)
)
(and
(g2 t1o13)
(f t1o15 t1o13)
)
(and
(g2 t1o14)
(f t1o15 t1o14)
)
(and
(g2 t1o15)
(f t1o15 t1o15)
)
(and
(g2 t1o16)
(f t1o15 t1o16)
)
)
)
(=>
(g1 t1o16)
(or
(and
(g2 t1o1)
(f t1o16 t1o1)
)
(and
(g2 t1o2)
(f t1o16 t1o2)
)
(and
(g2 t1o3)
(f t1o16 t1o3)
)
(and
(g2 t1o4)
(f t1o16 t1o4)
)
(and
(g2 t1o5)
(f t1o16 t1o5)
)
(and
(g2 t1o6)
(f t1o16 t1o6)
)
(and
(g2 t1o7)
(f t1o16 t1o7)
)
(and
(g2 t1o8)
(f t1o16 t1o8)
)
(and
(g2 t1o9)
(f t1o16 t1o9)
)
(and
(g2 t1o10)
(f t1o16 t1o10)
)
(and
(g2 t1o11)
(f t1o16 t1o11)
)
(and
(g2 t1o12)
(f t1o16 t1o12)
)
(and
(g2 t1o13)
(f t1o16 t1o13)
)
(and
(g2 t1o14)
(f t1o16 t1o14)
)
(and
(g2 t1o15)
(f t1o16 t1o15)
)
(and
(g2 t1o16)
(f t1o16 t1o16)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(or
(and
(g1 t1o1)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o1 t1o1)
)
(=>
(g2 t1o2)
(f t1o1 t1o2)
)
(=>
(g2 t1o3)
(f t1o1 t1o3)
)
(=>
(g2 t1o4)
(f t1o1 t1o4)
)
(=>
(g2 t1o5)
(f t1o1 t1o5)
)
(=>
(g2 t1o6)
(f t1o1 t1o6)
)
(=>
(g2 t1o7)
(f t1o1 t1o7)
)
(=>
(g2 t1o8)
(f t1o1 t1o8)
)
(=>
(g2 t1o9)
(f t1o1 t1o9)
)
(=>
(g2 t1o10)
(f t1o1 t1o10)
)
(=>
(g2 t1o11)
(f t1o1 t1o11)
)
(=>
(g2 t1o12)
(f t1o1 t1o12)
)
(=>
(g2 t1o13)
(f t1o1 t1o13)
)
(=>
(g2 t1o14)
(f t1o1 t1o14)
)
(=>
(g2 t1o15)
(f t1o1 t1o15)
)
(=>
(g2 t1o16)
(f t1o1 t1o16)
)
)
)
)
(and
(g1 t1o2)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o2 t1o1)
)
(=>
(g2 t1o2)
(f t1o2 t1o2)
)
(=>
(g2 t1o3)
(f t1o2 t1o3)
)
(=>
(g2 t1o4)
(f t1o2 t1o4)
)
(=>
(g2 t1o5)
(f t1o2 t1o5)
)
(=>
(g2 t1o6)
(f t1o2 t1o6)
)
(=>
(g2 t1o7)
(f t1o2 t1o7)
)
(=>
(g2 t1o8)
(f t1o2 t1o8)
)
(=>
(g2 t1o9)
(f t1o2 t1o9)
)
(=>
(g2 t1o10)
(f t1o2 t1o10)
)
(=>
(g2 t1o11)
(f t1o2 t1o11)
)
(=>
(g2 t1o12)
(f t1o2 t1o12)
)
(=>
(g2 t1o13)
(f t1o2 t1o13)
)
(=>
(g2 t1o14)
(f t1o2 t1o14)
)
(=>
(g2 t1o15)
(f t1o2 t1o15)
)
(=>
(g2 t1o16)
(f t1o2 t1o16)
)
)
)
)
(and
(g1 t1o3)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o3 t1o1)
)
(=>
(g2 t1o2)
(f t1o3 t1o2)
)
(=>
(g2 t1o3)
(f t1o3 t1o3)
)
(=>
(g2 t1o4)
(f t1o3 t1o4)
)
(=>
(g2 t1o5)
(f t1o3 t1o5)
)
(=>
(g2 t1o6)
(f t1o3 t1o6)
)
(=>
(g2 t1o7)
(f t1o3 t1o7)
)
(=>
(g2 t1o8)
(f t1o3 t1o8)
)
(=>
(g2 t1o9)
(f t1o3 t1o9)
)
(=>
(g2 t1o10)
(f t1o3 t1o10)
)
(=>
(g2 t1o11)
(f t1o3 t1o11)
)
(=>
(g2 t1o12)
(f t1o3 t1o12)
)
(=>
(g2 t1o13)
(f t1o3 t1o13)
)
(=>
(g2 t1o14)
(f t1o3 t1o14)
)
(=>
(g2 t1o15)
(f t1o3 t1o15)
)
(=>
(g2 t1o16)
(f t1o3 t1o16)
)
)
)
)
(and
(g1 t1o4)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o4 t1o1)
)
(=>
(g2 t1o2)
(f t1o4 t1o2)
)
(=>
(g2 t1o3)
(f t1o4 t1o3)
)
(=>
(g2 t1o4)
(f t1o4 t1o4)
)
(=>
(g2 t1o5)
(f t1o4 t1o5)
)
(=>
(g2 t1o6)
(f t1o4 t1o6)
)
(=>
(g2 t1o7)
(f t1o4 t1o7)
)
(=>
(g2 t1o8)
(f t1o4 t1o8)
)
(=>
(g2 t1o9)
(f t1o4 t1o9)
)
(=>
(g2 t1o10)
(f t1o4 t1o10)
)
(=>
(g2 t1o11)
(f t1o4 t1o11)
)
(=>
(g2 t1o12)
(f t1o4 t1o12)
)
(=>
(g2 t1o13)
(f t1o4 t1o13)
)
(=>
(g2 t1o14)
(f t1o4 t1o14)
)
(=>
(g2 t1o15)
(f t1o4 t1o15)
)
(=>
(g2 t1o16)
(f t1o4 t1o16)
)
)
)
)
(and
(g1 t1o5)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o5 t1o1)
)
(=>
(g2 t1o2)
(f t1o5 t1o2)
)
(=>
(g2 t1o3)
(f t1o5 t1o3)
)
(=>
(g2 t1o4)
(f t1o5 t1o4)
)
(=>
(g2 t1o5)
(f t1o5 t1o5)
)
(=>
(g2 t1o6)
(f t1o5 t1o6)
)
(=>
(g2 t1o7)
(f t1o5 t1o7)
)
(=>
(g2 t1o8)
(f t1o5 t1o8)
)
(=>
(g2 t1o9)
(f t1o5 t1o9)
)
(=>
(g2 t1o10)
(f t1o5 t1o10)
)
(=>
(g2 t1o11)
(f t1o5 t1o11)
)
(=>
(g2 t1o12)
(f t1o5 t1o12)
)
(=>
(g2 t1o13)
(f t1o5 t1o13)
)
(=>
(g2 t1o14)
(f t1o5 t1o14)
)
(=>
(g2 t1o15)
(f t1o5 t1o15)
)
(=>
(g2 t1o16)
(f t1o5 t1o16)
)
)
)
)
(and
(g1 t1o6)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o6 t1o1)
)
(=>
(g2 t1o2)
(f t1o6 t1o2)
)
(=>
(g2 t1o3)
(f t1o6 t1o3)
)
(=>
(g2 t1o4)
(f t1o6 t1o4)
)
(=>
(g2 t1o5)
(f t1o6 t1o5)
)
(=>
(g2 t1o6)
(f t1o6 t1o6)
)
(=>
(g2 t1o7)
(f t1o6 t1o7)
)
(=>
(g2 t1o8)
(f t1o6 t1o8)
)
(=>
(g2 t1o9)
(f t1o6 t1o9)
)
(=>
(g2 t1o10)
(f t1o6 t1o10)
)
(=>
(g2 t1o11)
(f t1o6 t1o11)
)
(=>
(g2 t1o12)
(f t1o6 t1o12)
)
(=>
(g2 t1o13)
(f t1o6 t1o13)
)
(=>
(g2 t1o14)
(f t1o6 t1o14)
)
(=>
(g2 t1o15)
(f t1o6 t1o15)
)
(=>
(g2 t1o16)
(f t1o6 t1o16)
)
)
)
)
(and
(g1 t1o7)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o7 t1o1)
)
(=>
(g2 t1o2)
(f t1o7 t1o2)
)
(=>
(g2 t1o3)
(f t1o7 t1o3)
)
(=>
(g2 t1o4)
(f t1o7 t1o4)
)
(=>
(g2 t1o5)
(f t1o7 t1o5)
)
(=>
(g2 t1o6)
(f t1o7 t1o6)
)
(=>
(g2 t1o7)
(f t1o7 t1o7)
)
(=>
(g2 t1o8)
(f t1o7 t1o8)
)
(=>
(g2 t1o9)
(f t1o7 t1o9)
)
(=>
(g2 t1o10)
(f t1o7 t1o10)
)
(=>
(g2 t1o11)
(f t1o7 t1o11)
)
(=>
(g2 t1o12)
(f t1o7 t1o12)
)
(=>
(g2 t1o13)
(f t1o7 t1o13)
)
(=>
(g2 t1o14)
(f t1o7 t1o14)
)
(=>
(g2 t1o15)
(f t1o7 t1o15)
)
(=>
(g2 t1o16)
(f t1o7 t1o16)
)
)
)
)
(and
(g1 t1o8)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o8 t1o1)
)
(=>
(g2 t1o2)
(f t1o8 t1o2)
)
(=>
(g2 t1o3)
(f t1o8 t1o3)
)
(=>
(g2 t1o4)
(f t1o8 t1o4)
)
(=>
(g2 t1o5)
(f t1o8 t1o5)
)
(=>
(g2 t1o6)
(f t1o8 t1o6)
)
(=>
(g2 t1o7)
(f t1o8 t1o7)
)
(=>
(g2 t1o8)
(f t1o8 t1o8)
)
(=>
(g2 t1o9)
(f t1o8 t1o9)
)
(=>
(g2 t1o10)
(f t1o8 t1o10)
)
(=>
(g2 t1o11)
(f t1o8 t1o11)
)
(=>
(g2 t1o12)
(f t1o8 t1o12)
)
(=>
(g2 t1o13)
(f t1o8 t1o13)
)
(=>
(g2 t1o14)
(f t1o8 t1o14)
)
(=>
(g2 t1o15)
(f t1o8 t1o15)
)
(=>
(g2 t1o16)
(f t1o8 t1o16)
)
)
)
)
(and
(g1 t1o9)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o9 t1o1)
)
(=>
(g2 t1o2)
(f t1o9 t1o2)
)
(=>
(g2 t1o3)
(f t1o9 t1o3)
)
(=>
(g2 t1o4)
(f t1o9 t1o4)
)
(=>
(g2 t1o5)
(f t1o9 t1o5)
)
(=>
(g2 t1o6)
(f t1o9 t1o6)
)
(=>
(g2 t1o7)
(f t1o9 t1o7)
)
(=>
(g2 t1o8)
(f t1o9 t1o8)
)
(=>
(g2 t1o9)
(f t1o9 t1o9)
)
(=>
(g2 t1o10)
(f t1o9 t1o10)
)
(=>
(g2 t1o11)
(f t1o9 t1o11)
)
(=>
(g2 t1o12)
(f t1o9 t1o12)
)
(=>
(g2 t1o13)
(f t1o9 t1o13)
)
(=>
(g2 t1o14)
(f t1o9 t1o14)
)
(=>
(g2 t1o15)
(f t1o9 t1o15)
)
(=>
(g2 t1o16)
(f t1o9 t1o16)
)
)
)
)
(and
(g1 t1o10)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o10 t1o1)
)
(=>
(g2 t1o2)
(f t1o10 t1o2)
)
(=>
(g2 t1o3)
(f t1o10 t1o3)
)
(=>
(g2 t1o4)
(f t1o10 t1o4)
)
(=>
(g2 t1o5)
(f t1o10 t1o5)
)
(=>
(g2 t1o6)
(f t1o10 t1o6)
)
(=>
(g2 t1o7)
(f t1o10 t1o7)
)
(=>
(g2 t1o8)
(f t1o10 t1o8)
)
(=>
(g2 t1o9)
(f t1o10 t1o9)
)
(=>
(g2 t1o10)
(f t1o10 t1o10)
)
(=>
(g2 t1o11)
(f t1o10 t1o11)
)
(=>
(g2 t1o12)
(f t1o10 t1o12)
)
(=>
(g2 t1o13)
(f t1o10 t1o13)
)
(=>
(g2 t1o14)
(f t1o10 t1o14)
)
(=>
(g2 t1o15)
(f t1o10 t1o15)
)
(=>
(g2 t1o16)
(f t1o10 t1o16)
)
)
)
)
(and
(g1 t1o11)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o11 t1o1)
)
(=>
(g2 t1o2)
(f t1o11 t1o2)
)
(=>
(g2 t1o3)
(f t1o11 t1o3)
)
(=>
(g2 t1o4)
(f t1o11 t1o4)
)
(=>
(g2 t1o5)
(f t1o11 t1o5)
)
(=>
(g2 t1o6)
(f t1o11 t1o6)
)
(=>
(g2 t1o7)
(f t1o11 t1o7)
)
(=>
(g2 t1o8)
(f t1o11 t1o8)
)
(=>
(g2 t1o9)
(f t1o11 t1o9)
)
(=>
(g2 t1o10)
(f t1o11 t1o10)
)
(=>
(g2 t1o11)
(f t1o11 t1o11)
)
(=>
(g2 t1o12)
(f t1o11 t1o12)
)
(=>
(g2 t1o13)
(f t1o11 t1o13)
)
(=>
(g2 t1o14)
(f t1o11 t1o14)
)
(=>
(g2 t1o15)
(f t1o11 t1o15)
)
(=>
(g2 t1o16)
(f t1o11 t1o16)
)
)
)
)
(and
(g1 t1o12)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o12 t1o1)
)
(=>
(g2 t1o2)
(f t1o12 t1o2)
)
(=>
(g2 t1o3)
(f t1o12 t1o3)
)
(=>
(g2 t1o4)
(f t1o12 t1o4)
)
(=>
(g2 t1o5)
(f t1o12 t1o5)
)
(=>
(g2 t1o6)
(f t1o12 t1o6)
)
(=>
(g2 t1o7)
(f t1o12 t1o7)
)
(=>
(g2 t1o8)
(f t1o12 t1o8)
)
(=>
(g2 t1o9)
(f t1o12 t1o9)
)
(=>
(g2 t1o10)
(f t1o12 t1o10)
)
(=>
(g2 t1o11)
(f t1o12 t1o11)
)
(=>
(g2 t1o12)
(f t1o12 t1o12)
)
(=>
(g2 t1o13)
(f t1o12 t1o13)
)
(=>
(g2 t1o14)
(f t1o12 t1o14)
)
(=>
(g2 t1o15)
(f t1o12 t1o15)
)
(=>
(g2 t1o16)
(f t1o12 t1o16)
)
)
)
)
(and
(g1 t1o13)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o13 t1o1)
)
(=>
(g2 t1o2)
(f t1o13 t1o2)
)
(=>
(g2 t1o3)
(f t1o13 t1o3)
)
(=>
(g2 t1o4)
(f t1o13 t1o4)
)
(=>
(g2 t1o5)
(f t1o13 t1o5)
)
(=>
(g2 t1o6)
(f t1o13 t1o6)
)
(=>
(g2 t1o7)
(f t1o13 t1o7)
)
(=>
(g2 t1o8)
(f t1o13 t1o8)
)
(=>
(g2 t1o9)
(f t1o13 t1o9)
)
(=>
(g2 t1o10)
(f t1o13 t1o10)
)
(=>
(g2 t1o11)
(f t1o13 t1o11)
)
(=>
(g2 t1o12)
(f t1o13 t1o12)
)
(=>
(g2 t1o13)
(f t1o13 t1o13)
)
(=>
(g2 t1o14)
(f t1o13 t1o14)
)
(=>
(g2 t1o15)
(f t1o13 t1o15)
)
(=>
(g2 t1o16)
(f t1o13 t1o16)
)
)
)
)
(and
(g1 t1o14)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o14 t1o1)
)
(=>
(g2 t1o2)
(f t1o14 t1o2)
)
(=>
(g2 t1o3)
(f t1o14 t1o3)
)
(=>
(g2 t1o4)
(f t1o14 t1o4)
)
(=>
(g2 t1o5)
(f t1o14 t1o5)
)
(=>
(g2 t1o6)
(f t1o14 t1o6)
)
(=>
(g2 t1o7)
(f t1o14 t1o7)
)
(=>
(g2 t1o8)
(f t1o14 t1o8)
)
(=>
(g2 t1o9)
(f t1o14 t1o9)
)
(=>
(g2 t1o10)
(f t1o14 t1o10)
)
(=>
(g2 t1o11)
(f t1o14 t1o11)
)
(=>
(g2 t1o12)
(f t1o14 t1o12)
)
(=>
(g2 t1o13)
(f t1o14 t1o13)
)
(=>
(g2 t1o14)
(f t1o14 t1o14)
)
(=>
(g2 t1o15)
(f t1o14 t1o15)
)
(=>
(g2 t1o16)
(f t1o14 t1o16)
)
)
)
)
(and
(g1 t1o15)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o15 t1o1)
)
(=>
(g2 t1o2)
(f t1o15 t1o2)
)
(=>
(g2 t1o3)
(f t1o15 t1o3)
)
(=>
(g2 t1o4)
(f t1o15 t1o4)
)
(=>
(g2 t1o5)
(f t1o15 t1o5)
)
(=>
(g2 t1o6)
(f t1o15 t1o6)
)
(=>
(g2 t1o7)
(f t1o15 t1o7)
)
(=>
(g2 t1o8)
(f t1o15 t1o8)
)
(=>
(g2 t1o9)
(f t1o15 t1o9)
)
(=>
(g2 t1o10)
(f t1o15 t1o10)
)
(=>
(g2 t1o11)
(f t1o15 t1o11)
)
(=>
(g2 t1o12)
(f t1o15 t1o12)
)
(=>
(g2 t1o13)
(f t1o15 t1o13)
)
(=>
(g2 t1o14)
(f t1o15 t1o14)
)
(=>
(g2 t1o15)
(f t1o15 t1o15)
)
(=>
(g2 t1o16)
(f t1o15 t1o16)
)
)
)
)
(and
(g1 t1o16)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
(g2 t1o4)
(g2 t1o5)
(g2 t1o6)
(g2 t1o7)
(g2 t1o8)
(g2 t1o9)
(g2 t1o10)
(g2 t1o11)
(g2 t1o12)
(g2 t1o13)
(g2 t1o14)
(g2 t1o15)
(g2 t1o16)
)
(and
(=>
(g2 t1o1)
(f t1o16 t1o1)
)
(=>
(g2 t1o2)
(f t1o16 t1o2)
)
(=>
(g2 t1o3)
(f t1o16 t1o3)
)
(=>
(g2 t1o4)
(f t1o16 t1o4)
)
(=>
(g2 t1o5)
(f t1o16 t1o5)
)
(=>
(g2 t1o6)
(f t1o16 t1o6)
)
(=>
(g2 t1o7)
(f t1o16 t1o7)
)
(=>
(g2 t1o8)
(f t1o16 t1o8)
)
(=>
(g2 t1o9)
(f t1o16 t1o9)
)
(=>
(g2 t1o10)
(f t1o16 t1o10)
)
(=>
(g2 t1o11)
(f t1o16 t1o11)
)
(=>
(g2 t1o12)
(f t1o16 t1o12)
)
(=>
(g2 t1o13)
(f t1o16 t1o13)
)
(=>
(g2 t1o14)
(f t1o16 t1o14)
)
(=>
(g2 t1o15)
(f t1o16 t1o15)
)
(=>
(g2 t1o16)
(f t1o16 t1o16)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(or
(and
(g1 t1o1)
(or
(and
(g2 t1o1)
(f t1o1 t1o1)
)
(and
(g2 t1o2)
(f t1o1 t1o2)
)
(and
(g2 t1o3)
(f t1o1 t1o3)
)
(and
(g2 t1o4)
(f t1o1 t1o4)
)
(and
(g2 t1o5)
(f t1o1 t1o5)
)
(and
(g2 t1o6)
(f t1o1 t1o6)
)
(and
(g2 t1o7)
(f t1o1 t1o7)
)
(and
(g2 t1o8)
(f t1o1 t1o8)
)
(and
(g2 t1o9)
(f t1o1 t1o9)
)
(and
(g2 t1o10)
(f t1o1 t1o10)
)
(and
(g2 t1o11)
(f t1o1 t1o11)
)
(and
(g2 t1o12)
(f t1o1 t1o12)
)
(and
(g2 t1o13)
(f t1o1 t1o13)
)
(and
(g2 t1o14)
(f t1o1 t1o14)
)
(and
(g2 t1o15)
(f t1o1 t1o15)
)
(and
(g2 t1o16)
(f t1o1 t1o16)
)
)
)
(and
(g1 t1o2)
(or
(and
(g2 t1o1)
(f t1o2 t1o1)
)
(and
(g2 t1o2)
(f t1o2 t1o2)
)
(and
(g2 t1o3)
(f t1o2 t1o3)
)
(and
(g2 t1o4)
(f t1o2 t1o4)
)
(and
(g2 t1o5)
(f t1o2 t1o5)
)
(and
(g2 t1o6)
(f t1o2 t1o6)
)
(and
(g2 t1o7)
(f t1o2 t1o7)
)
(and
(g2 t1o8)
(f t1o2 t1o8)
)
(and
(g2 t1o9)
(f t1o2 t1o9)
)
(and
(g2 t1o10)
(f t1o2 t1o10)
)
(and
(g2 t1o11)
(f t1o2 t1o11)
)
(and
(g2 t1o12)
(f t1o2 t1o12)
)
(and
(g2 t1o13)
(f t1o2 t1o13)
)
(and
(g2 t1o14)
(f t1o2 t1o14)
)
(and
(g2 t1o15)
(f t1o2 t1o15)
)
(and
(g2 t1o16)
(f t1o2 t1o16)
)
)
)
(and
(g1 t1o3)
(or
(and
(g2 t1o1)
(f t1o3 t1o1)
)
(and
(g2 t1o2)
(f t1o3 t1o2)
)
(and
(g2 t1o3)
(f t1o3 t1o3)
)
(and
(g2 t1o4)
(f t1o3 t1o4)
)
(and
(g2 t1o5)
(f t1o3 t1o5)
)
(and
(g2 t1o6)
(f t1o3 t1o6)
)
(and
(g2 t1o7)
(f t1o3 t1o7)
)
(and
(g2 t1o8)
(f t1o3 t1o8)
)
(and
(g2 t1o9)
(f t1o3 t1o9)
)
(and
(g2 t1o10)
(f t1o3 t1o10)
)
(and
(g2 t1o11)
(f t1o3 t1o11)
)
(and
(g2 t1o12)
(f t1o3 t1o12)
)
(and
(g2 t1o13)
(f t1o3 t1o13)
)
(and
(g2 t1o14)
(f t1o3 t1o14)
)
(and
(g2 t1o15)
(f t1o3 t1o15)
)
(and
(g2 t1o16)
(f t1o3 t1o16)
)
)
)
(and
(g1 t1o4)
(or
(and
(g2 t1o1)
(f t1o4 t1o1)
)
(and
(g2 t1o2)
(f t1o4 t1o2)
)
(and
(g2 t1o3)
(f t1o4 t1o3)
)
(and
(g2 t1o4)
(f t1o4 t1o4)
)
(and
(g2 t1o5)
(f t1o4 t1o5)
)
(and
(g2 t1o6)
(f t1o4 t1o6)
)
(and
(g2 t1o7)
(f t1o4 t1o7)
)
(and
(g2 t1o8)
(f t1o4 t1o8)
)
(and
(g2 t1o9)
(f t1o4 t1o9)
)
(and
(g2 t1o10)
(f t1o4 t1o10)
)
(and
(g2 t1o11)
(f t1o4 t1o11)
)
(and
(g2 t1o12)
(f t1o4 t1o12)
)
(and
(g2 t1o13)
(f t1o4 t1o13)
)
(and
(g2 t1o14)
(f t1o4 t1o14)
)
(and
(g2 t1o15)
(f t1o4 t1o15)
)
(and
(g2 t1o16)
(f t1o4 t1o16)
)
)
)
(and
(g1 t1o5)
(or
(and
(g2 t1o1)
(f t1o5 t1o1)
)
(and
(g2 t1o2)
(f t1o5 t1o2)
)
(and
(g2 t1o3)
(f t1o5 t1o3)
)
(and
(g2 t1o4)
(f t1o5 t1o4)
)
(and
(g2 t1o5)
(f t1o5 t1o5)
)
(and
(g2 t1o6)
(f t1o5 t1o6)
)
(and
(g2 t1o7)
(f t1o5 t1o7)
)
(and
(g2 t1o8)
(f t1o5 t1o8)
)
(and
(g2 t1o9)
(f t1o5 t1o9)
)
(and
(g2 t1o10)
(f t1o5 t1o10)
)
(and
(g2 t1o11)
(f t1o5 t1o11)
)
(and
(g2 t1o12)
(f t1o5 t1o12)
)
(and
(g2 t1o13)
(f t1o5 t1o13)
)
(and
(g2 t1o14)
(f t1o5 t1o14)
)
(and
(g2 t1o15)
(f t1o5 t1o15)
)
(and
(g2 t1o16)
(f t1o5 t1o16)
)
)
)
(and
(g1 t1o6)
(or
(and
(g2 t1o1)
(f t1o6 t1o1)
)
(and
(g2 t1o2)
(f t1o6 t1o2)
)
(and
(g2 t1o3)
(f t1o6 t1o3)
)
(and
(g2 t1o4)
(f t1o6 t1o4)
)
(and
(g2 t1o5)
(f t1o6 t1o5)
)
(and
(g2 t1o6)
(f t1o6 t1o6)
)
(and
(g2 t1o7)
(f t1o6 t1o7)
)
(and
(g2 t1o8)
(f t1o6 t1o8)
)
(and
(g2 t1o9)
(f t1o6 t1o9)
)
(and
(g2 t1o10)
(f t1o6 t1o10)
)
(and
(g2 t1o11)
(f t1o6 t1o11)
)
(and
(g2 t1o12)
(f t1o6 t1o12)
)
(and
(g2 t1o13)
(f t1o6 t1o13)
)
(and
(g2 t1o14)
(f t1o6 t1o14)
)
(and
(g2 t1o15)
(f t1o6 t1o15)
)
(and
(g2 t1o16)
(f t1o6 t1o16)
)
)
)
(and
(g1 t1o7)
(or
(and
(g2 t1o1)
(f t1o7 t1o1)
)
(and
(g2 t1o2)
(f t1o7 t1o2)
)
(and
(g2 t1o3)
(f t1o7 t1o3)
)
(and
(g2 t1o4)
(f t1o7 t1o4)
)
(and
(g2 t1o5)
(f t1o7 t1o5)
)
(and
(g2 t1o6)
(f t1o7 t1o6)
)
(and
(g2 t1o7)
(f t1o7 t1o7)
)
(and
(g2 t1o8)
(f t1o7 t1o8)
)
(and
(g2 t1o9)
(f t1o7 t1o9)
)
(and
(g2 t1o10)
(f t1o7 t1o10)
)
(and
(g2 t1o11)
(f t1o7 t1o11)
)
(and
(g2 t1o12)
(f t1o7 t1o12)
)
(and
(g2 t1o13)
(f t1o7 t1o13)
)
(and
(g2 t1o14)
(f t1o7 t1o14)
)
(and
(g2 t1o15)
(f t1o7 t1o15)
)
(and
(g2 t1o16)
(f t1o7 t1o16)
)
)
)
(and
(g1 t1o8)
(or
(and
(g2 t1o1)
(f t1o8 t1o1)
)
(and
(g2 t1o2)
(f t1o8 t1o2)
)
(and
(g2 t1o3)
(f t1o8 t1o3)
)
(and
(g2 t1o4)
(f t1o8 t1o4)
)
(and
(g2 t1o5)
(f t1o8 t1o5)
)
(and
(g2 t1o6)
(f t1o8 t1o6)
)
(and
(g2 t1o7)
(f t1o8 t1o7)
)
(and
(g2 t1o8)
(f t1o8 t1o8)
)
(and
(g2 t1o9)
(f t1o8 t1o9)
)
(and
(g2 t1o10)
(f t1o8 t1o10)
)
(and
(g2 t1o11)
(f t1o8 t1o11)
)
(and
(g2 t1o12)
(f t1o8 t1o12)
)
(and
(g2 t1o13)
(f t1o8 t1o13)
)
(and
(g2 t1o14)
(f t1o8 t1o14)
)
(and
(g2 t1o15)
(f t1o8 t1o15)
)
(and
(g2 t1o16)
(f t1o8 t1o16)
)
)
)
(and
(g1 t1o9)
(or
(and
(g2 t1o1)
(f t1o9 t1o1)
)
(and
(g2 t1o2)
(f t1o9 t1o2)
)
(and
(g2 t1o3)
(f t1o9 t1o3)
)
(and
(g2 t1o4)
(f t1o9 t1o4)
)
(and
(g2 t1o5)
(f t1o9 t1o5)
)
(and
(g2 t1o6)
(f t1o9 t1o6)
)
(and
(g2 t1o7)
(f t1o9 t1o7)
)
(and
(g2 t1o8)
(f t1o9 t1o8)
)
(and
(g2 t1o9)
(f t1o9 t1o9)
)
(and
(g2 t1o10)
(f t1o9 t1o10)
)
(and
(g2 t1o11)
(f t1o9 t1o11)
)
(and
(g2 t1o12)
(f t1o9 t1o12)
)
(and
(g2 t1o13)
(f t1o9 t1o13)
)
(and
(g2 t1o14)
(f t1o9 t1o14)
)
(and
(g2 t1o15)
(f t1o9 t1o15)
)
(and
(g2 t1o16)
(f t1o9 t1o16)
)
)
)
(and
(g1 t1o10)
(or
(and
(g2 t1o1)
(f t1o10 t1o1)
)
(and
(g2 t1o2)
(f t1o10 t1o2)
)
(and
(g2 t1o3)
(f t1o10 t1o3)
)
(and
(g2 t1o4)
(f t1o10 t1o4)
)
(and
(g2 t1o5)
(f t1o10 t1o5)
)
(and
(g2 t1o6)
(f t1o10 t1o6)
)
(and
(g2 t1o7)
(f t1o10 t1o7)
)
(and
(g2 t1o8)
(f t1o10 t1o8)
)
(and
(g2 t1o9)
(f t1o10 t1o9)
)
(and
(g2 t1o10)
(f t1o10 t1o10)
)
(and
(g2 t1o11)
(f t1o10 t1o11)
)
(and
(g2 t1o12)
(f t1o10 t1o12)
)
(and
(g2 t1o13)
(f t1o10 t1o13)
)
(and
(g2 t1o14)
(f t1o10 t1o14)
)
(and
(g2 t1o15)
(f t1o10 t1o15)
)
(and
(g2 t1o16)
(f t1o10 t1o16)
)
)
)
(and
(g1 t1o11)
(or
(and
(g2 t1o1)
(f t1o11 t1o1)
)
(and
(g2 t1o2)
(f t1o11 t1o2)
)
(and
(g2 t1o3)
(f t1o11 t1o3)
)
(and
(g2 t1o4)
(f t1o11 t1o4)
)
(and
(g2 t1o5)
(f t1o11 t1o5)
)
(and
(g2 t1o6)
(f t1o11 t1o6)
)
(and
(g2 t1o7)
(f t1o11 t1o7)
)
(and
(g2 t1o8)
(f t1o11 t1o8)
)
(and
(g2 t1o9)
(f t1o11 t1o9)
)
(and
(g2 t1o10)
(f t1o11 t1o10)
)
(and
(g2 t1o11)
(f t1o11 t1o11)
)
(and
(g2 t1o12)
(f t1o11 t1o12)
)
(and
(g2 t1o13)
(f t1o11 t1o13)
)
(and
(g2 t1o14)
(f t1o11 t1o14)
)
(and
(g2 t1o15)
(f t1o11 t1o15)
)
(and
(g2 t1o16)
(f t1o11 t1o16)
)
)
)
(and
(g1 t1o12)
(or
(and
(g2 t1o1)
(f t1o12 t1o1)
)
(and
(g2 t1o2)
(f t1o12 t1o2)
)
(and
(g2 t1o3)
(f t1o12 t1o3)
)
(and
(g2 t1o4)
(f t1o12 t1o4)
)
(and
(g2 t1o5)
(f t1o12 t1o5)
)
(and
(g2 t1o6)
(f t1o12 t1o6)
)
(and
(g2 t1o7)
(f t1o12 t1o7)
)
(and
(g2 t1o8)
(f t1o12 t1o8)
)
(and
(g2 t1o9)
(f t1o12 t1o9)
)
(and
(g2 t1o10)
(f t1o12 t1o10)
)
(and
(g2 t1o11)
(f t1o12 t1o11)
)
(and
(g2 t1o12)
(f t1o12 t1o12)
)
(and
(g2 t1o13)
(f t1o12 t1o13)
)
(and
(g2 t1o14)
(f t1o12 t1o14)
)
(and
(g2 t1o15)
(f t1o12 t1o15)
)
(and
(g2 t1o16)
(f t1o12 t1o16)
)
)
)
(and
(g1 t1o13)
(or
(and
(g2 t1o1)
(f t1o13 t1o1)
)
(and
(g2 t1o2)
(f t1o13 t1o2)
)
(and
(g2 t1o3)
(f t1o13 t1o3)
)
(and
(g2 t1o4)
(f t1o13 t1o4)
)
(and
(g2 t1o5)
(f t1o13 t1o5)
)
(and
(g2 t1o6)
(f t1o13 t1o6)
)
(and
(g2 t1o7)
(f t1o13 t1o7)
)
(and
(g2 t1o8)
(f t1o13 t1o8)
)
(and
(g2 t1o9)
(f t1o13 t1o9)
)
(and
(g2 t1o10)
(f t1o13 t1o10)
)
(and
(g2 t1o11)
(f t1o13 t1o11)
)
(and
(g2 t1o12)
(f t1o13 t1o12)
)
(and
(g2 t1o13)
(f t1o13 t1o13)
)
(and
(g2 t1o14)
(f t1o13 t1o14)
)
(and
(g2 t1o15)
(f t1o13 t1o15)
)
(and
(g2 t1o16)
(f t1o13 t1o16)
)
)
)
(and
(g1 t1o14)
(or
(and
(g2 t1o1)
(f t1o14 t1o1)
)
(and
(g2 t1o2)
(f t1o14 t1o2)
)
(and
(g2 t1o3)
(f t1o14 t1o3)
)
(and
(g2 t1o4)
(f t1o14 t1o4)
)
(and
(g2 t1o5)
(f t1o14 t1o5)
)
(and
(g2 t1o6)
(f t1o14 t1o6)
)
(and
(g2 t1o7)
(f t1o14 t1o7)
)
(and
(g2 t1o8)
(f t1o14 t1o8)
)
(and
(g2 t1o9)
(f t1o14 t1o9)
)
(and
(g2 t1o10)
(f t1o14 t1o10)
)
(and
(g2 t1o11)
(f t1o14 t1o11)
)
(and
(g2 t1o12)
(f t1o14 t1o12)
)
(and
(g2 t1o13)
(f t1o14 t1o13)
)
(and
(g2 t1o14)
(f t1o14 t1o14)
)
(and
(g2 t1o15)
(f t1o14 t1o15)
)
(and
(g2 t1o16)
(f t1o14 t1o16)
)
)
)
(and
(g1 t1o15)
(or
(and
(g2 t1o1)
(f t1o15 t1o1)
)
(and
(g2 t1o2)
(f t1o15 t1o2)
)
(and
(g2 t1o3)
(f t1o15 t1o3)
)
(and
(g2 t1o4)
(f t1o15 t1o4)
)
(and
(g2 t1o5)
(f t1o15 t1o5)
)
(and
(g2 t1o6)
(f t1o15 t1o6)
)
(and
(g2 t1o7)
(f t1o15 t1o7)
)
(and
(g2 t1o8)
(f t1o15 t1o8)
)
(and
(g2 t1o9)
(f t1o15 t1o9)
)
(and
(g2 t1o10)
(f t1o15 t1o10)
)
(and
(g2 t1o11)
(f t1o15 t1o11)
)
(and
(g2 t1o12)
(f t1o15 t1o12)
)
(and
(g2 t1o13)
(f t1o15 t1o13)
)
(and
(g2 t1o14)
(f t1o15 t1o14)
)
(and
(g2 t1o15)
(f t1o15 t1o15)
)
(and
(g2 t1o16)
(f t1o15 t1o16)
)
)
)
(and
(g1 t1o16)
(or
(and
(g2 t1o1)
(f t1o16 t1o1)
)
(and
(g2 t1o2)
(f t1o16 t1o2)
)
(and
(g2 t1o3)
(f t1o16 t1o3)
)
(and
(g2 t1o4)
(f t1o16 t1o4)
)
(and
(g2 t1o5)
(f t1o16 t1o5)
)
(and
(g2 t1o6)
(f t1o16 t1o6)
)
(and
(g2 t1o7)
(f t1o16 t1o7)
)
(and
(g2 t1o8)
(f t1o16 t1o8)
)
(and
(g2 t1o9)
(f t1o16 t1o9)
)
(and
(g2 t1o10)
(f t1o16 t1o10)
)
(and
(g2 t1o11)
(f t1o16 t1o11)
)
(and
(g2 t1o12)
(f t1o16 t1o12)
)
(and
(g2 t1o13)
(f t1o16 t1o13)
)
(and
(g2 t1o14)
(f t1o16 t1o14)
)
(and
(g2 t1o15)
(f t1o16 t1o15)
)
(and
(g2 t1o16)
(f t1o16 t1o16)
)
)
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
(= q2 true)
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
(g1 t2o10)
(g1 t2o11)
(g1 t2o12)
(g1 t2o13)
(g1 t2o14)
(g1 t2o15)
)
(and
(=>
(g1 t2o1)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o1 t2o1)
)
(=>
(g2 t2o2)
(f t2o1 t2o2)
)
(=>
(g2 t2o3)
(f t2o1 t2o3)
)
(=>
(g2 t2o4)
(f t2o1 t2o4)
)
(=>
(g2 t2o5)
(f t2o1 t2o5)
)
(=>
(g2 t2o6)
(f t2o1 t2o6)
)
(=>
(g2 t2o7)
(f t2o1 t2o7)
)
(=>
(g2 t2o8)
(f t2o1 t2o8)
)
(=>
(g2 t2o9)
(f t2o1 t2o9)
)
(=>
(g2 t2o10)
(f t2o1 t2o10)
)
(=>
(g2 t2o11)
(f t2o1 t2o11)
)
(=>
(g2 t2o12)
(f t2o1 t2o12)
)
(=>
(g2 t2o13)
(f t2o1 t2o13)
)
(=>
(g2 t2o14)
(f t2o1 t2o14)
)
(=>
(g2 t2o15)
(f t2o1 t2o15)
)
)
)
)
(=>
(g1 t2o2)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o2 t2o1)
)
(=>
(g2 t2o2)
(f t2o2 t2o2)
)
(=>
(g2 t2o3)
(f t2o2 t2o3)
)
(=>
(g2 t2o4)
(f t2o2 t2o4)
)
(=>
(g2 t2o5)
(f t2o2 t2o5)
)
(=>
(g2 t2o6)
(f t2o2 t2o6)
)
(=>
(g2 t2o7)
(f t2o2 t2o7)
)
(=>
(g2 t2o8)
(f t2o2 t2o8)
)
(=>
(g2 t2o9)
(f t2o2 t2o9)
)
(=>
(g2 t2o10)
(f t2o2 t2o10)
)
(=>
(g2 t2o11)
(f t2o2 t2o11)
)
(=>
(g2 t2o12)
(f t2o2 t2o12)
)
(=>
(g2 t2o13)
(f t2o2 t2o13)
)
(=>
(g2 t2o14)
(f t2o2 t2o14)
)
(=>
(g2 t2o15)
(f t2o2 t2o15)
)
)
)
)
(=>
(g1 t2o3)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o3 t2o1)
)
(=>
(g2 t2o2)
(f t2o3 t2o2)
)
(=>
(g2 t2o3)
(f t2o3 t2o3)
)
(=>
(g2 t2o4)
(f t2o3 t2o4)
)
(=>
(g2 t2o5)
(f t2o3 t2o5)
)
(=>
(g2 t2o6)
(f t2o3 t2o6)
)
(=>
(g2 t2o7)
(f t2o3 t2o7)
)
(=>
(g2 t2o8)
(f t2o3 t2o8)
)
(=>
(g2 t2o9)
(f t2o3 t2o9)
)
(=>
(g2 t2o10)
(f t2o3 t2o10)
)
(=>
(g2 t2o11)
(f t2o3 t2o11)
)
(=>
(g2 t2o12)
(f t2o3 t2o12)
)
(=>
(g2 t2o13)
(f t2o3 t2o13)
)
(=>
(g2 t2o14)
(f t2o3 t2o14)
)
(=>
(g2 t2o15)
(f t2o3 t2o15)
)
)
)
)
(=>
(g1 t2o4)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o4 t2o1)
)
(=>
(g2 t2o2)
(f t2o4 t2o2)
)
(=>
(g2 t2o3)
(f t2o4 t2o3)
)
(=>
(g2 t2o4)
(f t2o4 t2o4)
)
(=>
(g2 t2o5)
(f t2o4 t2o5)
)
(=>
(g2 t2o6)
(f t2o4 t2o6)
)
(=>
(g2 t2o7)
(f t2o4 t2o7)
)
(=>
(g2 t2o8)
(f t2o4 t2o8)
)
(=>
(g2 t2o9)
(f t2o4 t2o9)
)
(=>
(g2 t2o10)
(f t2o4 t2o10)
)
(=>
(g2 t2o11)
(f t2o4 t2o11)
)
(=>
(g2 t2o12)
(f t2o4 t2o12)
)
(=>
(g2 t2o13)
(f t2o4 t2o13)
)
(=>
(g2 t2o14)
(f t2o4 t2o14)
)
(=>
(g2 t2o15)
(f t2o4 t2o15)
)
)
)
)
(=>
(g1 t2o5)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o5 t2o1)
)
(=>
(g2 t2o2)
(f t2o5 t2o2)
)
(=>
(g2 t2o3)
(f t2o5 t2o3)
)
(=>
(g2 t2o4)
(f t2o5 t2o4)
)
(=>
(g2 t2o5)
(f t2o5 t2o5)
)
(=>
(g2 t2o6)
(f t2o5 t2o6)
)
(=>
(g2 t2o7)
(f t2o5 t2o7)
)
(=>
(g2 t2o8)
(f t2o5 t2o8)
)
(=>
(g2 t2o9)
(f t2o5 t2o9)
)
(=>
(g2 t2o10)
(f t2o5 t2o10)
)
(=>
(g2 t2o11)
(f t2o5 t2o11)
)
(=>
(g2 t2o12)
(f t2o5 t2o12)
)
(=>
(g2 t2o13)
(f t2o5 t2o13)
)
(=>
(g2 t2o14)
(f t2o5 t2o14)
)
(=>
(g2 t2o15)
(f t2o5 t2o15)
)
)
)
)
(=>
(g1 t2o6)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o6 t2o1)
)
(=>
(g2 t2o2)
(f t2o6 t2o2)
)
(=>
(g2 t2o3)
(f t2o6 t2o3)
)
(=>
(g2 t2o4)
(f t2o6 t2o4)
)
(=>
(g2 t2o5)
(f t2o6 t2o5)
)
(=>
(g2 t2o6)
(f t2o6 t2o6)
)
(=>
(g2 t2o7)
(f t2o6 t2o7)
)
(=>
(g2 t2o8)
(f t2o6 t2o8)
)
(=>
(g2 t2o9)
(f t2o6 t2o9)
)
(=>
(g2 t2o10)
(f t2o6 t2o10)
)
(=>
(g2 t2o11)
(f t2o6 t2o11)
)
(=>
(g2 t2o12)
(f t2o6 t2o12)
)
(=>
(g2 t2o13)
(f t2o6 t2o13)
)
(=>
(g2 t2o14)
(f t2o6 t2o14)
)
(=>
(g2 t2o15)
(f t2o6 t2o15)
)
)
)
)
(=>
(g1 t2o7)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o7 t2o1)
)
(=>
(g2 t2o2)
(f t2o7 t2o2)
)
(=>
(g2 t2o3)
(f t2o7 t2o3)
)
(=>
(g2 t2o4)
(f t2o7 t2o4)
)
(=>
(g2 t2o5)
(f t2o7 t2o5)
)
(=>
(g2 t2o6)
(f t2o7 t2o6)
)
(=>
(g2 t2o7)
(f t2o7 t2o7)
)
(=>
(g2 t2o8)
(f t2o7 t2o8)
)
(=>
(g2 t2o9)
(f t2o7 t2o9)
)
(=>
(g2 t2o10)
(f t2o7 t2o10)
)
(=>
(g2 t2o11)
(f t2o7 t2o11)
)
(=>
(g2 t2o12)
(f t2o7 t2o12)
)
(=>
(g2 t2o13)
(f t2o7 t2o13)
)
(=>
(g2 t2o14)
(f t2o7 t2o14)
)
(=>
(g2 t2o15)
(f t2o7 t2o15)
)
)
)
)
(=>
(g1 t2o8)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o8 t2o1)
)
(=>
(g2 t2o2)
(f t2o8 t2o2)
)
(=>
(g2 t2o3)
(f t2o8 t2o3)
)
(=>
(g2 t2o4)
(f t2o8 t2o4)
)
(=>
(g2 t2o5)
(f t2o8 t2o5)
)
(=>
(g2 t2o6)
(f t2o8 t2o6)
)
(=>
(g2 t2o7)
(f t2o8 t2o7)
)
(=>
(g2 t2o8)
(f t2o8 t2o8)
)
(=>
(g2 t2o9)
(f t2o8 t2o9)
)
(=>
(g2 t2o10)
(f t2o8 t2o10)
)
(=>
(g2 t2o11)
(f t2o8 t2o11)
)
(=>
(g2 t2o12)
(f t2o8 t2o12)
)
(=>
(g2 t2o13)
(f t2o8 t2o13)
)
(=>
(g2 t2o14)
(f t2o8 t2o14)
)
(=>
(g2 t2o15)
(f t2o8 t2o15)
)
)
)
)
(=>
(g1 t2o9)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o9 t2o1)
)
(=>
(g2 t2o2)
(f t2o9 t2o2)
)
(=>
(g2 t2o3)
(f t2o9 t2o3)
)
(=>
(g2 t2o4)
(f t2o9 t2o4)
)
(=>
(g2 t2o5)
(f t2o9 t2o5)
)
(=>
(g2 t2o6)
(f t2o9 t2o6)
)
(=>
(g2 t2o7)
(f t2o9 t2o7)
)
(=>
(g2 t2o8)
(f t2o9 t2o8)
)
(=>
(g2 t2o9)
(f t2o9 t2o9)
)
(=>
(g2 t2o10)
(f t2o9 t2o10)
)
(=>
(g2 t2o11)
(f t2o9 t2o11)
)
(=>
(g2 t2o12)
(f t2o9 t2o12)
)
(=>
(g2 t2o13)
(f t2o9 t2o13)
)
(=>
(g2 t2o14)
(f t2o9 t2o14)
)
(=>
(g2 t2o15)
(f t2o9 t2o15)
)
)
)
)
(=>
(g1 t2o10)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o10 t2o1)
)
(=>
(g2 t2o2)
(f t2o10 t2o2)
)
(=>
(g2 t2o3)
(f t2o10 t2o3)
)
(=>
(g2 t2o4)
(f t2o10 t2o4)
)
(=>
(g2 t2o5)
(f t2o10 t2o5)
)
(=>
(g2 t2o6)
(f t2o10 t2o6)
)
(=>
(g2 t2o7)
(f t2o10 t2o7)
)
(=>
(g2 t2o8)
(f t2o10 t2o8)
)
(=>
(g2 t2o9)
(f t2o10 t2o9)
)
(=>
(g2 t2o10)
(f t2o10 t2o10)
)
(=>
(g2 t2o11)
(f t2o10 t2o11)
)
(=>
(g2 t2o12)
(f t2o10 t2o12)
)
(=>
(g2 t2o13)
(f t2o10 t2o13)
)
(=>
(g2 t2o14)
(f t2o10 t2o14)
)
(=>
(g2 t2o15)
(f t2o10 t2o15)
)
)
)
)
(=>
(g1 t2o11)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o11 t2o1)
)
(=>
(g2 t2o2)
(f t2o11 t2o2)
)
(=>
(g2 t2o3)
(f t2o11 t2o3)
)
(=>
(g2 t2o4)
(f t2o11 t2o4)
)
(=>
(g2 t2o5)
(f t2o11 t2o5)
)
(=>
(g2 t2o6)
(f t2o11 t2o6)
)
(=>
(g2 t2o7)
(f t2o11 t2o7)
)
(=>
(g2 t2o8)
(f t2o11 t2o8)
)
(=>
(g2 t2o9)
(f t2o11 t2o9)
)
(=>
(g2 t2o10)
(f t2o11 t2o10)
)
(=>
(g2 t2o11)
(f t2o11 t2o11)
)
(=>
(g2 t2o12)
(f t2o11 t2o12)
)
(=>
(g2 t2o13)
(f t2o11 t2o13)
)
(=>
(g2 t2o14)
(f t2o11 t2o14)
)
(=>
(g2 t2o15)
(f t2o11 t2o15)
)
)
)
)
(=>
(g1 t2o12)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o12 t2o1)
)
(=>
(g2 t2o2)
(f t2o12 t2o2)
)
(=>
(g2 t2o3)
(f t2o12 t2o3)
)
(=>
(g2 t2o4)
(f t2o12 t2o4)
)
(=>
(g2 t2o5)
(f t2o12 t2o5)
)
(=>
(g2 t2o6)
(f t2o12 t2o6)
)
(=>
(g2 t2o7)
(f t2o12 t2o7)
)
(=>
(g2 t2o8)
(f t2o12 t2o8)
)
(=>
(g2 t2o9)
(f t2o12 t2o9)
)
(=>
(g2 t2o10)
(f t2o12 t2o10)
)
(=>
(g2 t2o11)
(f t2o12 t2o11)
)
(=>
(g2 t2o12)
(f t2o12 t2o12)
)
(=>
(g2 t2o13)
(f t2o12 t2o13)
)
(=>
(g2 t2o14)
(f t2o12 t2o14)
)
(=>
(g2 t2o15)
(f t2o12 t2o15)
)
)
)
)
(=>
(g1 t2o13)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o13 t2o1)
)
(=>
(g2 t2o2)
(f t2o13 t2o2)
)
(=>
(g2 t2o3)
(f t2o13 t2o3)
)
(=>
(g2 t2o4)
(f t2o13 t2o4)
)
(=>
(g2 t2o5)
(f t2o13 t2o5)
)
(=>
(g2 t2o6)
(f t2o13 t2o6)
)
(=>
(g2 t2o7)
(f t2o13 t2o7)
)
(=>
(g2 t2o8)
(f t2o13 t2o8)
)
(=>
(g2 t2o9)
(f t2o13 t2o9)
)
(=>
(g2 t2o10)
(f t2o13 t2o10)
)
(=>
(g2 t2o11)
(f t2o13 t2o11)
)
(=>
(g2 t2o12)
(f t2o13 t2o12)
)
(=>
(g2 t2o13)
(f t2o13 t2o13)
)
(=>
(g2 t2o14)
(f t2o13 t2o14)
)
(=>
(g2 t2o15)
(f t2o13 t2o15)
)
)
)
)
(=>
(g1 t2o14)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o14 t2o1)
)
(=>
(g2 t2o2)
(f t2o14 t2o2)
)
(=>
(g2 t2o3)
(f t2o14 t2o3)
)
(=>
(g2 t2o4)
(f t2o14 t2o4)
)
(=>
(g2 t2o5)
(f t2o14 t2o5)
)
(=>
(g2 t2o6)
(f t2o14 t2o6)
)
(=>
(g2 t2o7)
(f t2o14 t2o7)
)
(=>
(g2 t2o8)
(f t2o14 t2o8)
)
(=>
(g2 t2o9)
(f t2o14 t2o9)
)
(=>
(g2 t2o10)
(f t2o14 t2o10)
)
(=>
(g2 t2o11)
(f t2o14 t2o11)
)
(=>
(g2 t2o12)
(f t2o14 t2o12)
)
(=>
(g2 t2o13)
(f t2o14 t2o13)
)
(=>
(g2 t2o14)
(f t2o14 t2o14)
)
(=>
(g2 t2o15)
(f t2o14 t2o15)
)
)
)
)
(=>
(g1 t2o15)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o15 t2o1)
)
(=>
(g2 t2o2)
(f t2o15 t2o2)
)
(=>
(g2 t2o3)
(f t2o15 t2o3)
)
(=>
(g2 t2o4)
(f t2o15 t2o4)
)
(=>
(g2 t2o5)
(f t2o15 t2o5)
)
(=>
(g2 t2o6)
(f t2o15 t2o6)
)
(=>
(g2 t2o7)
(f t2o15 t2o7)
)
(=>
(g2 t2o8)
(f t2o15 t2o8)
)
(=>
(g2 t2o9)
(f t2o15 t2o9)
)
(=>
(g2 t2o10)
(f t2o15 t2o10)
)
(=>
(g2 t2o11)
(f t2o15 t2o11)
)
(=>
(g2 t2o12)
(f t2o15 t2o12)
)
(=>
(g2 t2o13)
(f t2o15 t2o13)
)
(=>
(g2 t2o14)
(f t2o15 t2o14)
)
(=>
(g2 t2o15)
(f t2o15 t2o15)
)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
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
(g1 t2o10)
(g1 t2o11)
(g1 t2o12)
(g1 t2o13)
(g1 t2o14)
(g1 t2o15)
)
(and
(=>
(g1 t2o1)
(or
(and
(g2 t2o1)
(f t2o1 t2o1)
)
(and
(g2 t2o2)
(f t2o1 t2o2)
)
(and
(g2 t2o3)
(f t2o1 t2o3)
)
(and
(g2 t2o4)
(f t2o1 t2o4)
)
(and
(g2 t2o5)
(f t2o1 t2o5)
)
(and
(g2 t2o6)
(f t2o1 t2o6)
)
(and
(g2 t2o7)
(f t2o1 t2o7)
)
(and
(g2 t2o8)
(f t2o1 t2o8)
)
(and
(g2 t2o9)
(f t2o1 t2o9)
)
(and
(g2 t2o10)
(f t2o1 t2o10)
)
(and
(g2 t2o11)
(f t2o1 t2o11)
)
(and
(g2 t2o12)
(f t2o1 t2o12)
)
(and
(g2 t2o13)
(f t2o1 t2o13)
)
(and
(g2 t2o14)
(f t2o1 t2o14)
)
(and
(g2 t2o15)
(f t2o1 t2o15)
)
)
)
(=>
(g1 t2o2)
(or
(and
(g2 t2o1)
(f t2o2 t2o1)
)
(and
(g2 t2o2)
(f t2o2 t2o2)
)
(and
(g2 t2o3)
(f t2o2 t2o3)
)
(and
(g2 t2o4)
(f t2o2 t2o4)
)
(and
(g2 t2o5)
(f t2o2 t2o5)
)
(and
(g2 t2o6)
(f t2o2 t2o6)
)
(and
(g2 t2o7)
(f t2o2 t2o7)
)
(and
(g2 t2o8)
(f t2o2 t2o8)
)
(and
(g2 t2o9)
(f t2o2 t2o9)
)
(and
(g2 t2o10)
(f t2o2 t2o10)
)
(and
(g2 t2o11)
(f t2o2 t2o11)
)
(and
(g2 t2o12)
(f t2o2 t2o12)
)
(and
(g2 t2o13)
(f t2o2 t2o13)
)
(and
(g2 t2o14)
(f t2o2 t2o14)
)
(and
(g2 t2o15)
(f t2o2 t2o15)
)
)
)
(=>
(g1 t2o3)
(or
(and
(g2 t2o1)
(f t2o3 t2o1)
)
(and
(g2 t2o2)
(f t2o3 t2o2)
)
(and
(g2 t2o3)
(f t2o3 t2o3)
)
(and
(g2 t2o4)
(f t2o3 t2o4)
)
(and
(g2 t2o5)
(f t2o3 t2o5)
)
(and
(g2 t2o6)
(f t2o3 t2o6)
)
(and
(g2 t2o7)
(f t2o3 t2o7)
)
(and
(g2 t2o8)
(f t2o3 t2o8)
)
(and
(g2 t2o9)
(f t2o3 t2o9)
)
(and
(g2 t2o10)
(f t2o3 t2o10)
)
(and
(g2 t2o11)
(f t2o3 t2o11)
)
(and
(g2 t2o12)
(f t2o3 t2o12)
)
(and
(g2 t2o13)
(f t2o3 t2o13)
)
(and
(g2 t2o14)
(f t2o3 t2o14)
)
(and
(g2 t2o15)
(f t2o3 t2o15)
)
)
)
(=>
(g1 t2o4)
(or
(and
(g2 t2o1)
(f t2o4 t2o1)
)
(and
(g2 t2o2)
(f t2o4 t2o2)
)
(and
(g2 t2o3)
(f t2o4 t2o3)
)
(and
(g2 t2o4)
(f t2o4 t2o4)
)
(and
(g2 t2o5)
(f t2o4 t2o5)
)
(and
(g2 t2o6)
(f t2o4 t2o6)
)
(and
(g2 t2o7)
(f t2o4 t2o7)
)
(and
(g2 t2o8)
(f t2o4 t2o8)
)
(and
(g2 t2o9)
(f t2o4 t2o9)
)
(and
(g2 t2o10)
(f t2o4 t2o10)
)
(and
(g2 t2o11)
(f t2o4 t2o11)
)
(and
(g2 t2o12)
(f t2o4 t2o12)
)
(and
(g2 t2o13)
(f t2o4 t2o13)
)
(and
(g2 t2o14)
(f t2o4 t2o14)
)
(and
(g2 t2o15)
(f t2o4 t2o15)
)
)
)
(=>
(g1 t2o5)
(or
(and
(g2 t2o1)
(f t2o5 t2o1)
)
(and
(g2 t2o2)
(f t2o5 t2o2)
)
(and
(g2 t2o3)
(f t2o5 t2o3)
)
(and
(g2 t2o4)
(f t2o5 t2o4)
)
(and
(g2 t2o5)
(f t2o5 t2o5)
)
(and
(g2 t2o6)
(f t2o5 t2o6)
)
(and
(g2 t2o7)
(f t2o5 t2o7)
)
(and
(g2 t2o8)
(f t2o5 t2o8)
)
(and
(g2 t2o9)
(f t2o5 t2o9)
)
(and
(g2 t2o10)
(f t2o5 t2o10)
)
(and
(g2 t2o11)
(f t2o5 t2o11)
)
(and
(g2 t2o12)
(f t2o5 t2o12)
)
(and
(g2 t2o13)
(f t2o5 t2o13)
)
(and
(g2 t2o14)
(f t2o5 t2o14)
)
(and
(g2 t2o15)
(f t2o5 t2o15)
)
)
)
(=>
(g1 t2o6)
(or
(and
(g2 t2o1)
(f t2o6 t2o1)
)
(and
(g2 t2o2)
(f t2o6 t2o2)
)
(and
(g2 t2o3)
(f t2o6 t2o3)
)
(and
(g2 t2o4)
(f t2o6 t2o4)
)
(and
(g2 t2o5)
(f t2o6 t2o5)
)
(and
(g2 t2o6)
(f t2o6 t2o6)
)
(and
(g2 t2o7)
(f t2o6 t2o7)
)
(and
(g2 t2o8)
(f t2o6 t2o8)
)
(and
(g2 t2o9)
(f t2o6 t2o9)
)
(and
(g2 t2o10)
(f t2o6 t2o10)
)
(and
(g2 t2o11)
(f t2o6 t2o11)
)
(and
(g2 t2o12)
(f t2o6 t2o12)
)
(and
(g2 t2o13)
(f t2o6 t2o13)
)
(and
(g2 t2o14)
(f t2o6 t2o14)
)
(and
(g2 t2o15)
(f t2o6 t2o15)
)
)
)
(=>
(g1 t2o7)
(or
(and
(g2 t2o1)
(f t2o7 t2o1)
)
(and
(g2 t2o2)
(f t2o7 t2o2)
)
(and
(g2 t2o3)
(f t2o7 t2o3)
)
(and
(g2 t2o4)
(f t2o7 t2o4)
)
(and
(g2 t2o5)
(f t2o7 t2o5)
)
(and
(g2 t2o6)
(f t2o7 t2o6)
)
(and
(g2 t2o7)
(f t2o7 t2o7)
)
(and
(g2 t2o8)
(f t2o7 t2o8)
)
(and
(g2 t2o9)
(f t2o7 t2o9)
)
(and
(g2 t2o10)
(f t2o7 t2o10)
)
(and
(g2 t2o11)
(f t2o7 t2o11)
)
(and
(g2 t2o12)
(f t2o7 t2o12)
)
(and
(g2 t2o13)
(f t2o7 t2o13)
)
(and
(g2 t2o14)
(f t2o7 t2o14)
)
(and
(g2 t2o15)
(f t2o7 t2o15)
)
)
)
(=>
(g1 t2o8)
(or
(and
(g2 t2o1)
(f t2o8 t2o1)
)
(and
(g2 t2o2)
(f t2o8 t2o2)
)
(and
(g2 t2o3)
(f t2o8 t2o3)
)
(and
(g2 t2o4)
(f t2o8 t2o4)
)
(and
(g2 t2o5)
(f t2o8 t2o5)
)
(and
(g2 t2o6)
(f t2o8 t2o6)
)
(and
(g2 t2o7)
(f t2o8 t2o7)
)
(and
(g2 t2o8)
(f t2o8 t2o8)
)
(and
(g2 t2o9)
(f t2o8 t2o9)
)
(and
(g2 t2o10)
(f t2o8 t2o10)
)
(and
(g2 t2o11)
(f t2o8 t2o11)
)
(and
(g2 t2o12)
(f t2o8 t2o12)
)
(and
(g2 t2o13)
(f t2o8 t2o13)
)
(and
(g2 t2o14)
(f t2o8 t2o14)
)
(and
(g2 t2o15)
(f t2o8 t2o15)
)
)
)
(=>
(g1 t2o9)
(or
(and
(g2 t2o1)
(f t2o9 t2o1)
)
(and
(g2 t2o2)
(f t2o9 t2o2)
)
(and
(g2 t2o3)
(f t2o9 t2o3)
)
(and
(g2 t2o4)
(f t2o9 t2o4)
)
(and
(g2 t2o5)
(f t2o9 t2o5)
)
(and
(g2 t2o6)
(f t2o9 t2o6)
)
(and
(g2 t2o7)
(f t2o9 t2o7)
)
(and
(g2 t2o8)
(f t2o9 t2o8)
)
(and
(g2 t2o9)
(f t2o9 t2o9)
)
(and
(g2 t2o10)
(f t2o9 t2o10)
)
(and
(g2 t2o11)
(f t2o9 t2o11)
)
(and
(g2 t2o12)
(f t2o9 t2o12)
)
(and
(g2 t2o13)
(f t2o9 t2o13)
)
(and
(g2 t2o14)
(f t2o9 t2o14)
)
(and
(g2 t2o15)
(f t2o9 t2o15)
)
)
)
(=>
(g1 t2o10)
(or
(and
(g2 t2o1)
(f t2o10 t2o1)
)
(and
(g2 t2o2)
(f t2o10 t2o2)
)
(and
(g2 t2o3)
(f t2o10 t2o3)
)
(and
(g2 t2o4)
(f t2o10 t2o4)
)
(and
(g2 t2o5)
(f t2o10 t2o5)
)
(and
(g2 t2o6)
(f t2o10 t2o6)
)
(and
(g2 t2o7)
(f t2o10 t2o7)
)
(and
(g2 t2o8)
(f t2o10 t2o8)
)
(and
(g2 t2o9)
(f t2o10 t2o9)
)
(and
(g2 t2o10)
(f t2o10 t2o10)
)
(and
(g2 t2o11)
(f t2o10 t2o11)
)
(and
(g2 t2o12)
(f t2o10 t2o12)
)
(and
(g2 t2o13)
(f t2o10 t2o13)
)
(and
(g2 t2o14)
(f t2o10 t2o14)
)
(and
(g2 t2o15)
(f t2o10 t2o15)
)
)
)
(=>
(g1 t2o11)
(or
(and
(g2 t2o1)
(f t2o11 t2o1)
)
(and
(g2 t2o2)
(f t2o11 t2o2)
)
(and
(g2 t2o3)
(f t2o11 t2o3)
)
(and
(g2 t2o4)
(f t2o11 t2o4)
)
(and
(g2 t2o5)
(f t2o11 t2o5)
)
(and
(g2 t2o6)
(f t2o11 t2o6)
)
(and
(g2 t2o7)
(f t2o11 t2o7)
)
(and
(g2 t2o8)
(f t2o11 t2o8)
)
(and
(g2 t2o9)
(f t2o11 t2o9)
)
(and
(g2 t2o10)
(f t2o11 t2o10)
)
(and
(g2 t2o11)
(f t2o11 t2o11)
)
(and
(g2 t2o12)
(f t2o11 t2o12)
)
(and
(g2 t2o13)
(f t2o11 t2o13)
)
(and
(g2 t2o14)
(f t2o11 t2o14)
)
(and
(g2 t2o15)
(f t2o11 t2o15)
)
)
)
(=>
(g1 t2o12)
(or
(and
(g2 t2o1)
(f t2o12 t2o1)
)
(and
(g2 t2o2)
(f t2o12 t2o2)
)
(and
(g2 t2o3)
(f t2o12 t2o3)
)
(and
(g2 t2o4)
(f t2o12 t2o4)
)
(and
(g2 t2o5)
(f t2o12 t2o5)
)
(and
(g2 t2o6)
(f t2o12 t2o6)
)
(and
(g2 t2o7)
(f t2o12 t2o7)
)
(and
(g2 t2o8)
(f t2o12 t2o8)
)
(and
(g2 t2o9)
(f t2o12 t2o9)
)
(and
(g2 t2o10)
(f t2o12 t2o10)
)
(and
(g2 t2o11)
(f t2o12 t2o11)
)
(and
(g2 t2o12)
(f t2o12 t2o12)
)
(and
(g2 t2o13)
(f t2o12 t2o13)
)
(and
(g2 t2o14)
(f t2o12 t2o14)
)
(and
(g2 t2o15)
(f t2o12 t2o15)
)
)
)
(=>
(g1 t2o13)
(or
(and
(g2 t2o1)
(f t2o13 t2o1)
)
(and
(g2 t2o2)
(f t2o13 t2o2)
)
(and
(g2 t2o3)
(f t2o13 t2o3)
)
(and
(g2 t2o4)
(f t2o13 t2o4)
)
(and
(g2 t2o5)
(f t2o13 t2o5)
)
(and
(g2 t2o6)
(f t2o13 t2o6)
)
(and
(g2 t2o7)
(f t2o13 t2o7)
)
(and
(g2 t2o8)
(f t2o13 t2o8)
)
(and
(g2 t2o9)
(f t2o13 t2o9)
)
(and
(g2 t2o10)
(f t2o13 t2o10)
)
(and
(g2 t2o11)
(f t2o13 t2o11)
)
(and
(g2 t2o12)
(f t2o13 t2o12)
)
(and
(g2 t2o13)
(f t2o13 t2o13)
)
(and
(g2 t2o14)
(f t2o13 t2o14)
)
(and
(g2 t2o15)
(f t2o13 t2o15)
)
)
)
(=>
(g1 t2o14)
(or
(and
(g2 t2o1)
(f t2o14 t2o1)
)
(and
(g2 t2o2)
(f t2o14 t2o2)
)
(and
(g2 t2o3)
(f t2o14 t2o3)
)
(and
(g2 t2o4)
(f t2o14 t2o4)
)
(and
(g2 t2o5)
(f t2o14 t2o5)
)
(and
(g2 t2o6)
(f t2o14 t2o6)
)
(and
(g2 t2o7)
(f t2o14 t2o7)
)
(and
(g2 t2o8)
(f t2o14 t2o8)
)
(and
(g2 t2o9)
(f t2o14 t2o9)
)
(and
(g2 t2o10)
(f t2o14 t2o10)
)
(and
(g2 t2o11)
(f t2o14 t2o11)
)
(and
(g2 t2o12)
(f t2o14 t2o12)
)
(and
(g2 t2o13)
(f t2o14 t2o13)
)
(and
(g2 t2o14)
(f t2o14 t2o14)
)
(and
(g2 t2o15)
(f t2o14 t2o15)
)
)
)
(=>
(g1 t2o15)
(or
(and
(g2 t2o1)
(f t2o15 t2o1)
)
(and
(g2 t2o2)
(f t2o15 t2o2)
)
(and
(g2 t2o3)
(f t2o15 t2o3)
)
(and
(g2 t2o4)
(f t2o15 t2o4)
)
(and
(g2 t2o5)
(f t2o15 t2o5)
)
(and
(g2 t2o6)
(f t2o15 t2o6)
)
(and
(g2 t2o7)
(f t2o15 t2o7)
)
(and
(g2 t2o8)
(f t2o15 t2o8)
)
(and
(g2 t2o9)
(f t2o15 t2o9)
)
(and
(g2 t2o10)
(f t2o15 t2o10)
)
(and
(g2 t2o11)
(f t2o15 t2o11)
)
(and
(g2 t2o12)
(f t2o15 t2o12)
)
(and
(g2 t2o13)
(f t2o15 t2o13)
)
(and
(g2 t2o14)
(f t2o15 t2o14)
)
(and
(g2 t2o15)
(f t2o15 t2o15)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(or
(and
(g1 t2o1)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o1 t2o1)
)
(=>
(g2 t2o2)
(f t2o1 t2o2)
)
(=>
(g2 t2o3)
(f t2o1 t2o3)
)
(=>
(g2 t2o4)
(f t2o1 t2o4)
)
(=>
(g2 t2o5)
(f t2o1 t2o5)
)
(=>
(g2 t2o6)
(f t2o1 t2o6)
)
(=>
(g2 t2o7)
(f t2o1 t2o7)
)
(=>
(g2 t2o8)
(f t2o1 t2o8)
)
(=>
(g2 t2o9)
(f t2o1 t2o9)
)
(=>
(g2 t2o10)
(f t2o1 t2o10)
)
(=>
(g2 t2o11)
(f t2o1 t2o11)
)
(=>
(g2 t2o12)
(f t2o1 t2o12)
)
(=>
(g2 t2o13)
(f t2o1 t2o13)
)
(=>
(g2 t2o14)
(f t2o1 t2o14)
)
(=>
(g2 t2o15)
(f t2o1 t2o15)
)
)
)
)
(and
(g1 t2o2)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o2 t2o1)
)
(=>
(g2 t2o2)
(f t2o2 t2o2)
)
(=>
(g2 t2o3)
(f t2o2 t2o3)
)
(=>
(g2 t2o4)
(f t2o2 t2o4)
)
(=>
(g2 t2o5)
(f t2o2 t2o5)
)
(=>
(g2 t2o6)
(f t2o2 t2o6)
)
(=>
(g2 t2o7)
(f t2o2 t2o7)
)
(=>
(g2 t2o8)
(f t2o2 t2o8)
)
(=>
(g2 t2o9)
(f t2o2 t2o9)
)
(=>
(g2 t2o10)
(f t2o2 t2o10)
)
(=>
(g2 t2o11)
(f t2o2 t2o11)
)
(=>
(g2 t2o12)
(f t2o2 t2o12)
)
(=>
(g2 t2o13)
(f t2o2 t2o13)
)
(=>
(g2 t2o14)
(f t2o2 t2o14)
)
(=>
(g2 t2o15)
(f t2o2 t2o15)
)
)
)
)
(and
(g1 t2o3)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o3 t2o1)
)
(=>
(g2 t2o2)
(f t2o3 t2o2)
)
(=>
(g2 t2o3)
(f t2o3 t2o3)
)
(=>
(g2 t2o4)
(f t2o3 t2o4)
)
(=>
(g2 t2o5)
(f t2o3 t2o5)
)
(=>
(g2 t2o6)
(f t2o3 t2o6)
)
(=>
(g2 t2o7)
(f t2o3 t2o7)
)
(=>
(g2 t2o8)
(f t2o3 t2o8)
)
(=>
(g2 t2o9)
(f t2o3 t2o9)
)
(=>
(g2 t2o10)
(f t2o3 t2o10)
)
(=>
(g2 t2o11)
(f t2o3 t2o11)
)
(=>
(g2 t2o12)
(f t2o3 t2o12)
)
(=>
(g2 t2o13)
(f t2o3 t2o13)
)
(=>
(g2 t2o14)
(f t2o3 t2o14)
)
(=>
(g2 t2o15)
(f t2o3 t2o15)
)
)
)
)
(and
(g1 t2o4)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o4 t2o1)
)
(=>
(g2 t2o2)
(f t2o4 t2o2)
)
(=>
(g2 t2o3)
(f t2o4 t2o3)
)
(=>
(g2 t2o4)
(f t2o4 t2o4)
)
(=>
(g2 t2o5)
(f t2o4 t2o5)
)
(=>
(g2 t2o6)
(f t2o4 t2o6)
)
(=>
(g2 t2o7)
(f t2o4 t2o7)
)
(=>
(g2 t2o8)
(f t2o4 t2o8)
)
(=>
(g2 t2o9)
(f t2o4 t2o9)
)
(=>
(g2 t2o10)
(f t2o4 t2o10)
)
(=>
(g2 t2o11)
(f t2o4 t2o11)
)
(=>
(g2 t2o12)
(f t2o4 t2o12)
)
(=>
(g2 t2o13)
(f t2o4 t2o13)
)
(=>
(g2 t2o14)
(f t2o4 t2o14)
)
(=>
(g2 t2o15)
(f t2o4 t2o15)
)
)
)
)
(and
(g1 t2o5)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o5 t2o1)
)
(=>
(g2 t2o2)
(f t2o5 t2o2)
)
(=>
(g2 t2o3)
(f t2o5 t2o3)
)
(=>
(g2 t2o4)
(f t2o5 t2o4)
)
(=>
(g2 t2o5)
(f t2o5 t2o5)
)
(=>
(g2 t2o6)
(f t2o5 t2o6)
)
(=>
(g2 t2o7)
(f t2o5 t2o7)
)
(=>
(g2 t2o8)
(f t2o5 t2o8)
)
(=>
(g2 t2o9)
(f t2o5 t2o9)
)
(=>
(g2 t2o10)
(f t2o5 t2o10)
)
(=>
(g2 t2o11)
(f t2o5 t2o11)
)
(=>
(g2 t2o12)
(f t2o5 t2o12)
)
(=>
(g2 t2o13)
(f t2o5 t2o13)
)
(=>
(g2 t2o14)
(f t2o5 t2o14)
)
(=>
(g2 t2o15)
(f t2o5 t2o15)
)
)
)
)
(and
(g1 t2o6)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o6 t2o1)
)
(=>
(g2 t2o2)
(f t2o6 t2o2)
)
(=>
(g2 t2o3)
(f t2o6 t2o3)
)
(=>
(g2 t2o4)
(f t2o6 t2o4)
)
(=>
(g2 t2o5)
(f t2o6 t2o5)
)
(=>
(g2 t2o6)
(f t2o6 t2o6)
)
(=>
(g2 t2o7)
(f t2o6 t2o7)
)
(=>
(g2 t2o8)
(f t2o6 t2o8)
)
(=>
(g2 t2o9)
(f t2o6 t2o9)
)
(=>
(g2 t2o10)
(f t2o6 t2o10)
)
(=>
(g2 t2o11)
(f t2o6 t2o11)
)
(=>
(g2 t2o12)
(f t2o6 t2o12)
)
(=>
(g2 t2o13)
(f t2o6 t2o13)
)
(=>
(g2 t2o14)
(f t2o6 t2o14)
)
(=>
(g2 t2o15)
(f t2o6 t2o15)
)
)
)
)
(and
(g1 t2o7)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o7 t2o1)
)
(=>
(g2 t2o2)
(f t2o7 t2o2)
)
(=>
(g2 t2o3)
(f t2o7 t2o3)
)
(=>
(g2 t2o4)
(f t2o7 t2o4)
)
(=>
(g2 t2o5)
(f t2o7 t2o5)
)
(=>
(g2 t2o6)
(f t2o7 t2o6)
)
(=>
(g2 t2o7)
(f t2o7 t2o7)
)
(=>
(g2 t2o8)
(f t2o7 t2o8)
)
(=>
(g2 t2o9)
(f t2o7 t2o9)
)
(=>
(g2 t2o10)
(f t2o7 t2o10)
)
(=>
(g2 t2o11)
(f t2o7 t2o11)
)
(=>
(g2 t2o12)
(f t2o7 t2o12)
)
(=>
(g2 t2o13)
(f t2o7 t2o13)
)
(=>
(g2 t2o14)
(f t2o7 t2o14)
)
(=>
(g2 t2o15)
(f t2o7 t2o15)
)
)
)
)
(and
(g1 t2o8)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o8 t2o1)
)
(=>
(g2 t2o2)
(f t2o8 t2o2)
)
(=>
(g2 t2o3)
(f t2o8 t2o3)
)
(=>
(g2 t2o4)
(f t2o8 t2o4)
)
(=>
(g2 t2o5)
(f t2o8 t2o5)
)
(=>
(g2 t2o6)
(f t2o8 t2o6)
)
(=>
(g2 t2o7)
(f t2o8 t2o7)
)
(=>
(g2 t2o8)
(f t2o8 t2o8)
)
(=>
(g2 t2o9)
(f t2o8 t2o9)
)
(=>
(g2 t2o10)
(f t2o8 t2o10)
)
(=>
(g2 t2o11)
(f t2o8 t2o11)
)
(=>
(g2 t2o12)
(f t2o8 t2o12)
)
(=>
(g2 t2o13)
(f t2o8 t2o13)
)
(=>
(g2 t2o14)
(f t2o8 t2o14)
)
(=>
(g2 t2o15)
(f t2o8 t2o15)
)
)
)
)
(and
(g1 t2o9)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o9 t2o1)
)
(=>
(g2 t2o2)
(f t2o9 t2o2)
)
(=>
(g2 t2o3)
(f t2o9 t2o3)
)
(=>
(g2 t2o4)
(f t2o9 t2o4)
)
(=>
(g2 t2o5)
(f t2o9 t2o5)
)
(=>
(g2 t2o6)
(f t2o9 t2o6)
)
(=>
(g2 t2o7)
(f t2o9 t2o7)
)
(=>
(g2 t2o8)
(f t2o9 t2o8)
)
(=>
(g2 t2o9)
(f t2o9 t2o9)
)
(=>
(g2 t2o10)
(f t2o9 t2o10)
)
(=>
(g2 t2o11)
(f t2o9 t2o11)
)
(=>
(g2 t2o12)
(f t2o9 t2o12)
)
(=>
(g2 t2o13)
(f t2o9 t2o13)
)
(=>
(g2 t2o14)
(f t2o9 t2o14)
)
(=>
(g2 t2o15)
(f t2o9 t2o15)
)
)
)
)
(and
(g1 t2o10)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o10 t2o1)
)
(=>
(g2 t2o2)
(f t2o10 t2o2)
)
(=>
(g2 t2o3)
(f t2o10 t2o3)
)
(=>
(g2 t2o4)
(f t2o10 t2o4)
)
(=>
(g2 t2o5)
(f t2o10 t2o5)
)
(=>
(g2 t2o6)
(f t2o10 t2o6)
)
(=>
(g2 t2o7)
(f t2o10 t2o7)
)
(=>
(g2 t2o8)
(f t2o10 t2o8)
)
(=>
(g2 t2o9)
(f t2o10 t2o9)
)
(=>
(g2 t2o10)
(f t2o10 t2o10)
)
(=>
(g2 t2o11)
(f t2o10 t2o11)
)
(=>
(g2 t2o12)
(f t2o10 t2o12)
)
(=>
(g2 t2o13)
(f t2o10 t2o13)
)
(=>
(g2 t2o14)
(f t2o10 t2o14)
)
(=>
(g2 t2o15)
(f t2o10 t2o15)
)
)
)
)
(and
(g1 t2o11)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o11 t2o1)
)
(=>
(g2 t2o2)
(f t2o11 t2o2)
)
(=>
(g2 t2o3)
(f t2o11 t2o3)
)
(=>
(g2 t2o4)
(f t2o11 t2o4)
)
(=>
(g2 t2o5)
(f t2o11 t2o5)
)
(=>
(g2 t2o6)
(f t2o11 t2o6)
)
(=>
(g2 t2o7)
(f t2o11 t2o7)
)
(=>
(g2 t2o8)
(f t2o11 t2o8)
)
(=>
(g2 t2o9)
(f t2o11 t2o9)
)
(=>
(g2 t2o10)
(f t2o11 t2o10)
)
(=>
(g2 t2o11)
(f t2o11 t2o11)
)
(=>
(g2 t2o12)
(f t2o11 t2o12)
)
(=>
(g2 t2o13)
(f t2o11 t2o13)
)
(=>
(g2 t2o14)
(f t2o11 t2o14)
)
(=>
(g2 t2o15)
(f t2o11 t2o15)
)
)
)
)
(and
(g1 t2o12)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o12 t2o1)
)
(=>
(g2 t2o2)
(f t2o12 t2o2)
)
(=>
(g2 t2o3)
(f t2o12 t2o3)
)
(=>
(g2 t2o4)
(f t2o12 t2o4)
)
(=>
(g2 t2o5)
(f t2o12 t2o5)
)
(=>
(g2 t2o6)
(f t2o12 t2o6)
)
(=>
(g2 t2o7)
(f t2o12 t2o7)
)
(=>
(g2 t2o8)
(f t2o12 t2o8)
)
(=>
(g2 t2o9)
(f t2o12 t2o9)
)
(=>
(g2 t2o10)
(f t2o12 t2o10)
)
(=>
(g2 t2o11)
(f t2o12 t2o11)
)
(=>
(g2 t2o12)
(f t2o12 t2o12)
)
(=>
(g2 t2o13)
(f t2o12 t2o13)
)
(=>
(g2 t2o14)
(f t2o12 t2o14)
)
(=>
(g2 t2o15)
(f t2o12 t2o15)
)
)
)
)
(and
(g1 t2o13)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o13 t2o1)
)
(=>
(g2 t2o2)
(f t2o13 t2o2)
)
(=>
(g2 t2o3)
(f t2o13 t2o3)
)
(=>
(g2 t2o4)
(f t2o13 t2o4)
)
(=>
(g2 t2o5)
(f t2o13 t2o5)
)
(=>
(g2 t2o6)
(f t2o13 t2o6)
)
(=>
(g2 t2o7)
(f t2o13 t2o7)
)
(=>
(g2 t2o8)
(f t2o13 t2o8)
)
(=>
(g2 t2o9)
(f t2o13 t2o9)
)
(=>
(g2 t2o10)
(f t2o13 t2o10)
)
(=>
(g2 t2o11)
(f t2o13 t2o11)
)
(=>
(g2 t2o12)
(f t2o13 t2o12)
)
(=>
(g2 t2o13)
(f t2o13 t2o13)
)
(=>
(g2 t2o14)
(f t2o13 t2o14)
)
(=>
(g2 t2o15)
(f t2o13 t2o15)
)
)
)
)
(and
(g1 t2o14)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o14 t2o1)
)
(=>
(g2 t2o2)
(f t2o14 t2o2)
)
(=>
(g2 t2o3)
(f t2o14 t2o3)
)
(=>
(g2 t2o4)
(f t2o14 t2o4)
)
(=>
(g2 t2o5)
(f t2o14 t2o5)
)
(=>
(g2 t2o6)
(f t2o14 t2o6)
)
(=>
(g2 t2o7)
(f t2o14 t2o7)
)
(=>
(g2 t2o8)
(f t2o14 t2o8)
)
(=>
(g2 t2o9)
(f t2o14 t2o9)
)
(=>
(g2 t2o10)
(f t2o14 t2o10)
)
(=>
(g2 t2o11)
(f t2o14 t2o11)
)
(=>
(g2 t2o12)
(f t2o14 t2o12)
)
(=>
(g2 t2o13)
(f t2o14 t2o13)
)
(=>
(g2 t2o14)
(f t2o14 t2o14)
)
(=>
(g2 t2o15)
(f t2o14 t2o15)
)
)
)
)
(and
(g1 t2o15)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
(g2 t2o4)
(g2 t2o5)
(g2 t2o6)
(g2 t2o7)
(g2 t2o8)
(g2 t2o9)
(g2 t2o10)
(g2 t2o11)
(g2 t2o12)
(g2 t2o13)
(g2 t2o14)
(g2 t2o15)
)
(and
(=>
(g2 t2o1)
(f t2o15 t2o1)
)
(=>
(g2 t2o2)
(f t2o15 t2o2)
)
(=>
(g2 t2o3)
(f t2o15 t2o3)
)
(=>
(g2 t2o4)
(f t2o15 t2o4)
)
(=>
(g2 t2o5)
(f t2o15 t2o5)
)
(=>
(g2 t2o6)
(f t2o15 t2o6)
)
(=>
(g2 t2o7)
(f t2o15 t2o7)
)
(=>
(g2 t2o8)
(f t2o15 t2o8)
)
(=>
(g2 t2o9)
(f t2o15 t2o9)
)
(=>
(g2 t2o10)
(f t2o15 t2o10)
)
(=>
(g2 t2o11)
(f t2o15 t2o11)
)
(=>
(g2 t2o12)
(f t2o15 t2o12)
)
(=>
(g2 t2o13)
(f t2o15 t2o13)
)
(=>
(g2 t2o14)
(f t2o15 t2o14)
)
(=>
(g2 t2o15)
(f t2o15 t2o15)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(or
(and
(g1 t2o1)
(or
(and
(g2 t2o1)
(f t2o1 t2o1)
)
(and
(g2 t2o2)
(f t2o1 t2o2)
)
(and
(g2 t2o3)
(f t2o1 t2o3)
)
(and
(g2 t2o4)
(f t2o1 t2o4)
)
(and
(g2 t2o5)
(f t2o1 t2o5)
)
(and
(g2 t2o6)
(f t2o1 t2o6)
)
(and
(g2 t2o7)
(f t2o1 t2o7)
)
(and
(g2 t2o8)
(f t2o1 t2o8)
)
(and
(g2 t2o9)
(f t2o1 t2o9)
)
(and
(g2 t2o10)
(f t2o1 t2o10)
)
(and
(g2 t2o11)
(f t2o1 t2o11)
)
(and
(g2 t2o12)
(f t2o1 t2o12)
)
(and
(g2 t2o13)
(f t2o1 t2o13)
)
(and
(g2 t2o14)
(f t2o1 t2o14)
)
(and
(g2 t2o15)
(f t2o1 t2o15)
)
)
)
(and
(g1 t2o2)
(or
(and
(g2 t2o1)
(f t2o2 t2o1)
)
(and
(g2 t2o2)
(f t2o2 t2o2)
)
(and
(g2 t2o3)
(f t2o2 t2o3)
)
(and
(g2 t2o4)
(f t2o2 t2o4)
)
(and
(g2 t2o5)
(f t2o2 t2o5)
)
(and
(g2 t2o6)
(f t2o2 t2o6)
)
(and
(g2 t2o7)
(f t2o2 t2o7)
)
(and
(g2 t2o8)
(f t2o2 t2o8)
)
(and
(g2 t2o9)
(f t2o2 t2o9)
)
(and
(g2 t2o10)
(f t2o2 t2o10)
)
(and
(g2 t2o11)
(f t2o2 t2o11)
)
(and
(g2 t2o12)
(f t2o2 t2o12)
)
(and
(g2 t2o13)
(f t2o2 t2o13)
)
(and
(g2 t2o14)
(f t2o2 t2o14)
)
(and
(g2 t2o15)
(f t2o2 t2o15)
)
)
)
(and
(g1 t2o3)
(or
(and
(g2 t2o1)
(f t2o3 t2o1)
)
(and
(g2 t2o2)
(f t2o3 t2o2)
)
(and
(g2 t2o3)
(f t2o3 t2o3)
)
(and
(g2 t2o4)
(f t2o3 t2o4)
)
(and
(g2 t2o5)
(f t2o3 t2o5)
)
(and
(g2 t2o6)
(f t2o3 t2o6)
)
(and
(g2 t2o7)
(f t2o3 t2o7)
)
(and
(g2 t2o8)
(f t2o3 t2o8)
)
(and
(g2 t2o9)
(f t2o3 t2o9)
)
(and
(g2 t2o10)
(f t2o3 t2o10)
)
(and
(g2 t2o11)
(f t2o3 t2o11)
)
(and
(g2 t2o12)
(f t2o3 t2o12)
)
(and
(g2 t2o13)
(f t2o3 t2o13)
)
(and
(g2 t2o14)
(f t2o3 t2o14)
)
(and
(g2 t2o15)
(f t2o3 t2o15)
)
)
)
(and
(g1 t2o4)
(or
(and
(g2 t2o1)
(f t2o4 t2o1)
)
(and
(g2 t2o2)
(f t2o4 t2o2)
)
(and
(g2 t2o3)
(f t2o4 t2o3)
)
(and
(g2 t2o4)
(f t2o4 t2o4)
)
(and
(g2 t2o5)
(f t2o4 t2o5)
)
(and
(g2 t2o6)
(f t2o4 t2o6)
)
(and
(g2 t2o7)
(f t2o4 t2o7)
)
(and
(g2 t2o8)
(f t2o4 t2o8)
)
(and
(g2 t2o9)
(f t2o4 t2o9)
)
(and
(g2 t2o10)
(f t2o4 t2o10)
)
(and
(g2 t2o11)
(f t2o4 t2o11)
)
(and
(g2 t2o12)
(f t2o4 t2o12)
)
(and
(g2 t2o13)
(f t2o4 t2o13)
)
(and
(g2 t2o14)
(f t2o4 t2o14)
)
(and
(g2 t2o15)
(f t2o4 t2o15)
)
)
)
(and
(g1 t2o5)
(or
(and
(g2 t2o1)
(f t2o5 t2o1)
)
(and
(g2 t2o2)
(f t2o5 t2o2)
)
(and
(g2 t2o3)
(f t2o5 t2o3)
)
(and
(g2 t2o4)
(f t2o5 t2o4)
)
(and
(g2 t2o5)
(f t2o5 t2o5)
)
(and
(g2 t2o6)
(f t2o5 t2o6)
)
(and
(g2 t2o7)
(f t2o5 t2o7)
)
(and
(g2 t2o8)
(f t2o5 t2o8)
)
(and
(g2 t2o9)
(f t2o5 t2o9)
)
(and
(g2 t2o10)
(f t2o5 t2o10)
)
(and
(g2 t2o11)
(f t2o5 t2o11)
)
(and
(g2 t2o12)
(f t2o5 t2o12)
)
(and
(g2 t2o13)
(f t2o5 t2o13)
)
(and
(g2 t2o14)
(f t2o5 t2o14)
)
(and
(g2 t2o15)
(f t2o5 t2o15)
)
)
)
(and
(g1 t2o6)
(or
(and
(g2 t2o1)
(f t2o6 t2o1)
)
(and
(g2 t2o2)
(f t2o6 t2o2)
)
(and
(g2 t2o3)
(f t2o6 t2o3)
)
(and
(g2 t2o4)
(f t2o6 t2o4)
)
(and
(g2 t2o5)
(f t2o6 t2o5)
)
(and
(g2 t2o6)
(f t2o6 t2o6)
)
(and
(g2 t2o7)
(f t2o6 t2o7)
)
(and
(g2 t2o8)
(f t2o6 t2o8)
)
(and
(g2 t2o9)
(f t2o6 t2o9)
)
(and
(g2 t2o10)
(f t2o6 t2o10)
)
(and
(g2 t2o11)
(f t2o6 t2o11)
)
(and
(g2 t2o12)
(f t2o6 t2o12)
)
(and
(g2 t2o13)
(f t2o6 t2o13)
)
(and
(g2 t2o14)
(f t2o6 t2o14)
)
(and
(g2 t2o15)
(f t2o6 t2o15)
)
)
)
(and
(g1 t2o7)
(or
(and
(g2 t2o1)
(f t2o7 t2o1)
)
(and
(g2 t2o2)
(f t2o7 t2o2)
)
(and
(g2 t2o3)
(f t2o7 t2o3)
)
(and
(g2 t2o4)
(f t2o7 t2o4)
)
(and
(g2 t2o5)
(f t2o7 t2o5)
)
(and
(g2 t2o6)
(f t2o7 t2o6)
)
(and
(g2 t2o7)
(f t2o7 t2o7)
)
(and
(g2 t2o8)
(f t2o7 t2o8)
)
(and
(g2 t2o9)
(f t2o7 t2o9)
)
(and
(g2 t2o10)
(f t2o7 t2o10)
)
(and
(g2 t2o11)
(f t2o7 t2o11)
)
(and
(g2 t2o12)
(f t2o7 t2o12)
)
(and
(g2 t2o13)
(f t2o7 t2o13)
)
(and
(g2 t2o14)
(f t2o7 t2o14)
)
(and
(g2 t2o15)
(f t2o7 t2o15)
)
)
)
(and
(g1 t2o8)
(or
(and
(g2 t2o1)
(f t2o8 t2o1)
)
(and
(g2 t2o2)
(f t2o8 t2o2)
)
(and
(g2 t2o3)
(f t2o8 t2o3)
)
(and
(g2 t2o4)
(f t2o8 t2o4)
)
(and
(g2 t2o5)
(f t2o8 t2o5)
)
(and
(g2 t2o6)
(f t2o8 t2o6)
)
(and
(g2 t2o7)
(f t2o8 t2o7)
)
(and
(g2 t2o8)
(f t2o8 t2o8)
)
(and
(g2 t2o9)
(f t2o8 t2o9)
)
(and
(g2 t2o10)
(f t2o8 t2o10)
)
(and
(g2 t2o11)
(f t2o8 t2o11)
)
(and
(g2 t2o12)
(f t2o8 t2o12)
)
(and
(g2 t2o13)
(f t2o8 t2o13)
)
(and
(g2 t2o14)
(f t2o8 t2o14)
)
(and
(g2 t2o15)
(f t2o8 t2o15)
)
)
)
(and
(g1 t2o9)
(or
(and
(g2 t2o1)
(f t2o9 t2o1)
)
(and
(g2 t2o2)
(f t2o9 t2o2)
)
(and
(g2 t2o3)
(f t2o9 t2o3)
)
(and
(g2 t2o4)
(f t2o9 t2o4)
)
(and
(g2 t2o5)
(f t2o9 t2o5)
)
(and
(g2 t2o6)
(f t2o9 t2o6)
)
(and
(g2 t2o7)
(f t2o9 t2o7)
)
(and
(g2 t2o8)
(f t2o9 t2o8)
)
(and
(g2 t2o9)
(f t2o9 t2o9)
)
(and
(g2 t2o10)
(f t2o9 t2o10)
)
(and
(g2 t2o11)
(f t2o9 t2o11)
)
(and
(g2 t2o12)
(f t2o9 t2o12)
)
(and
(g2 t2o13)
(f t2o9 t2o13)
)
(and
(g2 t2o14)
(f t2o9 t2o14)
)
(and
(g2 t2o15)
(f t2o9 t2o15)
)
)
)
(and
(g1 t2o10)
(or
(and
(g2 t2o1)
(f t2o10 t2o1)
)
(and
(g2 t2o2)
(f t2o10 t2o2)
)
(and
(g2 t2o3)
(f t2o10 t2o3)
)
(and
(g2 t2o4)
(f t2o10 t2o4)
)
(and
(g2 t2o5)
(f t2o10 t2o5)
)
(and
(g2 t2o6)
(f t2o10 t2o6)
)
(and
(g2 t2o7)
(f t2o10 t2o7)
)
(and
(g2 t2o8)
(f t2o10 t2o8)
)
(and
(g2 t2o9)
(f t2o10 t2o9)
)
(and
(g2 t2o10)
(f t2o10 t2o10)
)
(and
(g2 t2o11)
(f t2o10 t2o11)
)
(and
(g2 t2o12)
(f t2o10 t2o12)
)
(and
(g2 t2o13)
(f t2o10 t2o13)
)
(and
(g2 t2o14)
(f t2o10 t2o14)
)
(and
(g2 t2o15)
(f t2o10 t2o15)
)
)
)
(and
(g1 t2o11)
(or
(and
(g2 t2o1)
(f t2o11 t2o1)
)
(and
(g2 t2o2)
(f t2o11 t2o2)
)
(and
(g2 t2o3)
(f t2o11 t2o3)
)
(and
(g2 t2o4)
(f t2o11 t2o4)
)
(and
(g2 t2o5)
(f t2o11 t2o5)
)
(and
(g2 t2o6)
(f t2o11 t2o6)
)
(and
(g2 t2o7)
(f t2o11 t2o7)
)
(and
(g2 t2o8)
(f t2o11 t2o8)
)
(and
(g2 t2o9)
(f t2o11 t2o9)
)
(and
(g2 t2o10)
(f t2o11 t2o10)
)
(and
(g2 t2o11)
(f t2o11 t2o11)
)
(and
(g2 t2o12)
(f t2o11 t2o12)
)
(and
(g2 t2o13)
(f t2o11 t2o13)
)
(and
(g2 t2o14)
(f t2o11 t2o14)
)
(and
(g2 t2o15)
(f t2o11 t2o15)
)
)
)
(and
(g1 t2o12)
(or
(and
(g2 t2o1)
(f t2o12 t2o1)
)
(and
(g2 t2o2)
(f t2o12 t2o2)
)
(and
(g2 t2o3)
(f t2o12 t2o3)
)
(and
(g2 t2o4)
(f t2o12 t2o4)
)
(and
(g2 t2o5)
(f t2o12 t2o5)
)
(and
(g2 t2o6)
(f t2o12 t2o6)
)
(and
(g2 t2o7)
(f t2o12 t2o7)
)
(and
(g2 t2o8)
(f t2o12 t2o8)
)
(and
(g2 t2o9)
(f t2o12 t2o9)
)
(and
(g2 t2o10)
(f t2o12 t2o10)
)
(and
(g2 t2o11)
(f t2o12 t2o11)
)
(and
(g2 t2o12)
(f t2o12 t2o12)
)
(and
(g2 t2o13)
(f t2o12 t2o13)
)
(and
(g2 t2o14)
(f t2o12 t2o14)
)
(and
(g2 t2o15)
(f t2o12 t2o15)
)
)
)
(and
(g1 t2o13)
(or
(and
(g2 t2o1)
(f t2o13 t2o1)
)
(and
(g2 t2o2)
(f t2o13 t2o2)
)
(and
(g2 t2o3)
(f t2o13 t2o3)
)
(and
(g2 t2o4)
(f t2o13 t2o4)
)
(and
(g2 t2o5)
(f t2o13 t2o5)
)
(and
(g2 t2o6)
(f t2o13 t2o6)
)
(and
(g2 t2o7)
(f t2o13 t2o7)
)
(and
(g2 t2o8)
(f t2o13 t2o8)
)
(and
(g2 t2o9)
(f t2o13 t2o9)
)
(and
(g2 t2o10)
(f t2o13 t2o10)
)
(and
(g2 t2o11)
(f t2o13 t2o11)
)
(and
(g2 t2o12)
(f t2o13 t2o12)
)
(and
(g2 t2o13)
(f t2o13 t2o13)
)
(and
(g2 t2o14)
(f t2o13 t2o14)
)
(and
(g2 t2o15)
(f t2o13 t2o15)
)
)
)
(and
(g1 t2o14)
(or
(and
(g2 t2o1)
(f t2o14 t2o1)
)
(and
(g2 t2o2)
(f t2o14 t2o2)
)
(and
(g2 t2o3)
(f t2o14 t2o3)
)
(and
(g2 t2o4)
(f t2o14 t2o4)
)
(and
(g2 t2o5)
(f t2o14 t2o5)
)
(and
(g2 t2o6)
(f t2o14 t2o6)
)
(and
(g2 t2o7)
(f t2o14 t2o7)
)
(and
(g2 t2o8)
(f t2o14 t2o8)
)
(and
(g2 t2o9)
(f t2o14 t2o9)
)
(and
(g2 t2o10)
(f t2o14 t2o10)
)
(and
(g2 t2o11)
(f t2o14 t2o11)
)
(and
(g2 t2o12)
(f t2o14 t2o12)
)
(and
(g2 t2o13)
(f t2o14 t2o13)
)
(and
(g2 t2o14)
(f t2o14 t2o14)
)
(and
(g2 t2o15)
(f t2o14 t2o15)
)
)
)
(and
(g1 t2o15)
(or
(and
(g2 t2o1)
(f t2o15 t2o1)
)
(and
(g2 t2o2)
(f t2o15 t2o2)
)
(and
(g2 t2o3)
(f t2o15 t2o3)
)
(and
(g2 t2o4)
(f t2o15 t2o4)
)
(and
(g2 t2o5)
(f t2o15 t2o5)
)
(and
(g2 t2o6)
(f t2o15 t2o6)
)
(and
(g2 t2o7)
(f t2o15 t2o7)
)
(and
(g2 t2o8)
(f t2o15 t2o8)
)
(and
(g2 t2o9)
(f t2o15 t2o9)
)
(and
(g2 t2o10)
(f t2o15 t2o10)
)
(and
(g2 t2o11)
(f t2o15 t2o11)
)
(and
(g2 t2o12)
(f t2o15 t2o12)
)
(and
(g2 t2o13)
(f t2o15 t2o13)
)
(and
(g2 t2o14)
(f t2o15 t2o14)
)
(and
(g2 t2o15)
(f t2o15 t2o15)
)
)
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
(= q2 true)
)
(and
(or
(g1 t3o1)
(g1 t3o2)
)
(and
(=>
(g1 t3o1)
(and
(or
(g2 t3o1)
(g2 t3o2)
)
(and
(=>
(g2 t3o1)
(f t3o1 t3o1)
)
(=>
(g2 t3o2)
(f t3o1 t3o2)
)
)
)
)
(=>
(g1 t3o2)
(and
(or
(g2 t3o1)
(g2 t3o2)
)
(and
(=>
(g2 t3o1)
(f t3o2 t3o1)
)
(=>
(g2 t3o2)
(f t3o2 t3o2)
)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(and
(or
(g1 t3o1)
(g1 t3o2)
)
(and
(=>
(g1 t3o1)
(or
(and
(g2 t3o1)
(f t3o1 t3o1)
)
(and
(g2 t3o2)
(f t3o1 t3o2)
)
)
)
(=>
(g1 t3o2)
(or
(and
(g2 t3o1)
(f t3o2 t3o1)
)
(and
(g2 t3o2)
(f t3o2 t3o2)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(or
(and
(g1 t3o1)
(and
(or
(g2 t3o1)
(g2 t3o2)
)
(and
(=>
(g2 t3o1)
(f t3o1 t3o1)
)
(=>
(g2 t3o2)
(f t3o1 t3o2)
)
)
)
)
(and
(g1 t3o2)
(and
(or
(g2 t3o1)
(g2 t3o2)
)
(and
(=>
(g2 t3o1)
(f t3o2 t3o1)
)
(=>
(g2 t3o2)
(f t3o2 t3o2)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(or
(and
(g1 t3o1)
(or
(and
(g2 t3o1)
(f t3o1 t3o1)
)
(and
(g2 t3o2)
(f t3o1 t3o2)
)
)
)
(and
(g1 t3o2)
(or
(and
(g2 t3o1)
(f t3o2 t3o1)
)
(and
(g2 t3o2)
(f t3o2 t3o2)
)
)
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
(= q2 true)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
(g1 c1o3)
(g1 c1o4)
(g1 c1o5)
)
(and
(=>
(g1 c1o1)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o1 c1o1)
)
(=>
(g2 c1o2)
(f c1o1 c1o2)
)
(=>
(g2 c1o3)
(f c1o1 c1o3)
)
(=>
(g2 c1o4)
(f c1o1 c1o4)
)
(=>
(g2 c1o5)
(f c1o1 c1o5)
)
)
)
)
(=>
(g1 c1o2)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o2 c1o1)
)
(=>
(g2 c1o2)
(f c1o2 c1o2)
)
(=>
(g2 c1o3)
(f c1o2 c1o3)
)
(=>
(g2 c1o4)
(f c1o2 c1o4)
)
(=>
(g2 c1o5)
(f c1o2 c1o5)
)
)
)
)
(=>
(g1 c1o3)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o3 c1o1)
)
(=>
(g2 c1o2)
(f c1o3 c1o2)
)
(=>
(g2 c1o3)
(f c1o3 c1o3)
)
(=>
(g2 c1o4)
(f c1o3 c1o4)
)
(=>
(g2 c1o5)
(f c1o3 c1o5)
)
)
)
)
(=>
(g1 c1o4)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o4 c1o1)
)
(=>
(g2 c1o2)
(f c1o4 c1o2)
)
(=>
(g2 c1o3)
(f c1o4 c1o3)
)
(=>
(g2 c1o4)
(f c1o4 c1o4)
)
(=>
(g2 c1o5)
(f c1o4 c1o5)
)
)
)
)
(=>
(g1 c1o5)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o5 c1o1)
)
(=>
(g2 c1o2)
(f c1o5 c1o2)
)
(=>
(g2 c1o3)
(f c1o5 c1o3)
)
(=>
(g2 c1o4)
(f c1o5 c1o4)
)
(=>
(g2 c1o5)
(f c1o5 c1o5)
)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
(g1 c1o3)
(g1 c1o4)
(g1 c1o5)
)
(and
(=>
(g1 c1o1)
(or
(and
(g2 c1o1)
(f c1o1 c1o1)
)
(and
(g2 c1o2)
(f c1o1 c1o2)
)
(and
(g2 c1o3)
(f c1o1 c1o3)
)
(and
(g2 c1o4)
(f c1o1 c1o4)
)
(and
(g2 c1o5)
(f c1o1 c1o5)
)
)
)
(=>
(g1 c1o2)
(or
(and
(g2 c1o1)
(f c1o2 c1o1)
)
(and
(g2 c1o2)
(f c1o2 c1o2)
)
(and
(g2 c1o3)
(f c1o2 c1o3)
)
(and
(g2 c1o4)
(f c1o2 c1o4)
)
(and
(g2 c1o5)
(f c1o2 c1o5)
)
)
)
(=>
(g1 c1o3)
(or
(and
(g2 c1o1)
(f c1o3 c1o1)
)
(and
(g2 c1o2)
(f c1o3 c1o2)
)
(and
(g2 c1o3)
(f c1o3 c1o3)
)
(and
(g2 c1o4)
(f c1o3 c1o4)
)
(and
(g2 c1o5)
(f c1o3 c1o5)
)
)
)
(=>
(g1 c1o4)
(or
(and
(g2 c1o1)
(f c1o4 c1o1)
)
(and
(g2 c1o2)
(f c1o4 c1o2)
)
(and
(g2 c1o3)
(f c1o4 c1o3)
)
(and
(g2 c1o4)
(f c1o4 c1o4)
)
(and
(g2 c1o5)
(f c1o4 c1o5)
)
)
)
(=>
(g1 c1o5)
(or
(and
(g2 c1o1)
(f c1o5 c1o1)
)
(and
(g2 c1o2)
(f c1o5 c1o2)
)
(and
(g2 c1o3)
(f c1o5 c1o3)
)
(and
(g2 c1o4)
(f c1o5 c1o4)
)
(and
(g2 c1o5)
(f c1o5 c1o5)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(or
(and
(g1 c1o1)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o1 c1o1)
)
(=>
(g2 c1o2)
(f c1o1 c1o2)
)
(=>
(g2 c1o3)
(f c1o1 c1o3)
)
(=>
(g2 c1o4)
(f c1o1 c1o4)
)
(=>
(g2 c1o5)
(f c1o1 c1o5)
)
)
)
)
(and
(g1 c1o2)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o2 c1o1)
)
(=>
(g2 c1o2)
(f c1o2 c1o2)
)
(=>
(g2 c1o3)
(f c1o2 c1o3)
)
(=>
(g2 c1o4)
(f c1o2 c1o4)
)
(=>
(g2 c1o5)
(f c1o2 c1o5)
)
)
)
)
(and
(g1 c1o3)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o3 c1o1)
)
(=>
(g2 c1o2)
(f c1o3 c1o2)
)
(=>
(g2 c1o3)
(f c1o3 c1o3)
)
(=>
(g2 c1o4)
(f c1o3 c1o4)
)
(=>
(g2 c1o5)
(f c1o3 c1o5)
)
)
)
)
(and
(g1 c1o4)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o4 c1o1)
)
(=>
(g2 c1o2)
(f c1o4 c1o2)
)
(=>
(g2 c1o3)
(f c1o4 c1o3)
)
(=>
(g2 c1o4)
(f c1o4 c1o4)
)
(=>
(g2 c1o5)
(f c1o4 c1o5)
)
)
)
)
(and
(g1 c1o5)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
)
(and
(=>
(g2 c1o1)
(f c1o5 c1o1)
)
(=>
(g2 c1o2)
(f c1o5 c1o2)
)
(=>
(g2 c1o3)
(f c1o5 c1o3)
)
(=>
(g2 c1o4)
(f c1o5 c1o4)
)
(=>
(g2 c1o5)
(f c1o5 c1o5)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(or
(and
(g1 c1o1)
(or
(and
(g2 c1o1)
(f c1o1 c1o1)
)
(and
(g2 c1o2)
(f c1o1 c1o2)
)
(and
(g2 c1o3)
(f c1o1 c1o3)
)
(and
(g2 c1o4)
(f c1o1 c1o4)
)
(and
(g2 c1o5)
(f c1o1 c1o5)
)
)
)
(and
(g1 c1o2)
(or
(and
(g2 c1o1)
(f c1o2 c1o1)
)
(and
(g2 c1o2)
(f c1o2 c1o2)
)
(and
(g2 c1o3)
(f c1o2 c1o3)
)
(and
(g2 c1o4)
(f c1o2 c1o4)
)
(and
(g2 c1o5)
(f c1o2 c1o5)
)
)
)
(and
(g1 c1o3)
(or
(and
(g2 c1o1)
(f c1o3 c1o1)
)
(and
(g2 c1o2)
(f c1o3 c1o2)
)
(and
(g2 c1o3)
(f c1o3 c1o3)
)
(and
(g2 c1o4)
(f c1o3 c1o4)
)
(and
(g2 c1o5)
(f c1o3 c1o5)
)
)
)
(and
(g1 c1o4)
(or
(and
(g2 c1o1)
(f c1o4 c1o1)
)
(and
(g2 c1o2)
(f c1o4 c1o2)
)
(and
(g2 c1o3)
(f c1o4 c1o3)
)
(and
(g2 c1o4)
(f c1o4 c1o4)
)
(and
(g2 c1o5)
(f c1o4 c1o5)
)
)
)
(and
(g1 c1o5)
(or
(and
(g2 c1o1)
(f c1o5 c1o1)
)
(and
(g2 c1o2)
(f c1o5 c1o2)
)
(and
(g2 c1o3)
(f c1o5 c1o3)
)
(and
(g2 c1o4)
(f c1o5 c1o4)
)
(and
(g2 c1o5)
(f c1o5 c1o5)
)
)
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
(= q2 true)
)
(or
(and
(g1 c1o1)
(or
(and
(g2 c1o1)
(not
(f c1o1 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o1 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o1 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o1 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o1 c1o5)
)
)
)
)
(and
(g1 c1o2)
(or
(and
(g2 c1o1)
(not
(f c1o2 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o2 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o2 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o2 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o2 c1o5)
)
)
)
)
(and
(g1 c1o3)
(or
(and
(g2 c1o1)
(not
(f c1o3 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o3 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o3 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o3 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o3 c1o5)
)
)
)
)
(and
(g1 c1o4)
(or
(and
(g2 c1o1)
(not
(f c1o4 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o4 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o4 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o4 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o4 c1o5)
)
)
)
)
(and
(g1 c1o5)
(or
(and
(g2 c1o1)
(not
(f c1o5 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o5 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o5 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o5 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o5 c1o5)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(or
(and
(g1 c1o1)
(and
(=>
(g2 c1o1)
(not
(f c1o1 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o1 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o1 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o1 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o1 c1o5)
)
)
)
)
(and
(g1 c1o2)
(and
(=>
(g2 c1o1)
(not
(f c1o2 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o2 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o2 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o2 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o2 c1o5)
)
)
)
)
(and
(g1 c1o3)
(and
(=>
(g2 c1o1)
(not
(f c1o3 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o3 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o3 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o3 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o3 c1o5)
)
)
)
)
(and
(g1 c1o4)
(and
(=>
(g2 c1o1)
(not
(f c1o4 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o4 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o4 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o4 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o4 c1o5)
)
)
)
)
(and
(g1 c1o5)
(and
(=>
(g2 c1o1)
(not
(f c1o5 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o5 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o5 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o5 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o5 c1o5)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(and
(=>
(g1 c1o1)
(or
(and
(g2 c1o1)
(not
(f c1o1 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o1 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o1 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o1 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o1 c1o5)
)
)
)
)
(=>
(g1 c1o2)
(or
(and
(g2 c1o1)
(not
(f c1o2 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o2 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o2 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o2 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o2 c1o5)
)
)
)
)
(=>
(g1 c1o3)
(or
(and
(g2 c1o1)
(not
(f c1o3 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o3 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o3 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o3 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o3 c1o5)
)
)
)
)
(=>
(g1 c1o4)
(or
(and
(g2 c1o1)
(not
(f c1o4 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o4 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o4 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o4 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o4 c1o5)
)
)
)
)
(=>
(g1 c1o5)
(or
(and
(g2 c1o1)
(not
(f c1o5 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o5 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o5 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o5 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o5 c1o5)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(and
(=>
(g1 c1o1)
(and
(=>
(g2 c1o1)
(not
(f c1o1 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o1 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o1 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o1 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o1 c1o5)
)
)
)
)
(=>
(g1 c1o2)
(and
(=>
(g2 c1o1)
(not
(f c1o2 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o2 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o2 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o2 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o2 c1o5)
)
)
)
)
(=>
(g1 c1o3)
(and
(=>
(g2 c1o1)
(not
(f c1o3 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o3 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o3 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o3 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o3 c1o5)
)
)
)
)
(=>
(g1 c1o4)
(and
(=>
(g2 c1o1)
(not
(f c1o4 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o4 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o4 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o4 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o4 c1o5)
)
)
)
)
(=>
(g1 c1o5)
(and
(=>
(g2 c1o1)
(not
(f c1o5 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o5 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o5 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o5 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o5 c1o5)
)
)
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
(= q2 true)
)
(and
(or
(g1 c2o1)
(g1 c2o2)
)
(and
(=>
(g1 c2o1)
(and
(or
(g2 c2o1)
(g2 c2o2)
)
(and
(=>
(g2 c2o1)
(f c2o1 c2o1)
)
(=>
(g2 c2o2)
(f c2o1 c2o2)
)
)
)
)
(=>
(g1 c2o2)
(and
(or
(g2 c2o1)
(g2 c2o2)
)
(and
(=>
(g2 c2o1)
(f c2o2 c2o1)
)
(=>
(g2 c2o2)
(f c2o2 c2o2)
)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(and
(or
(g1 c2o1)
(g1 c2o2)
)
(and
(=>
(g1 c2o1)
(or
(and
(g2 c2o1)
(f c2o1 c2o1)
)
(and
(g2 c2o2)
(f c2o1 c2o2)
)
)
)
(=>
(g1 c2o2)
(or
(and
(g2 c2o1)
(f c2o2 c2o1)
)
(and
(g2 c2o2)
(f c2o2 c2o2)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(or
(and
(g1 c2o1)
(and
(or
(g2 c2o1)
(g2 c2o2)
)
(and
(=>
(g2 c2o1)
(f c2o1 c2o1)
)
(=>
(g2 c2o2)
(f c2o1 c2o2)
)
)
)
)
(and
(g1 c2o2)
(and
(or
(g2 c2o1)
(g2 c2o2)
)
(and
(=>
(g2 c2o1)
(f c2o2 c2o1)
)
(=>
(g2 c2o2)
(f c2o2 c2o2)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(or
(and
(g1 c2o1)
(or
(and
(g2 c2o1)
(f c2o1 c2o1)
)
(and
(g2 c2o2)
(f c2o1 c2o2)
)
)
)
(and
(g1 c2o2)
(or
(and
(g2 c2o1)
(f c2o2 c2o1)
)
(and
(g2 c2o2)
(f c2o2 c2o2)
)
)
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
(= q2 true)
)
(or
(and
(g1 c2o1)
(or
(and
(g2 c2o1)
(not
(f c2o1 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o1 c2o2)
)
)
)
)
(and
(g1 c2o2)
(or
(and
(g2 c2o1)
(not
(f c2o2 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o2 c2o2)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(or
(and
(g1 c2o1)
(and
(=>
(g2 c2o1)
(not
(f c2o1 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o1 c2o2)
)
)
)
)
(and
(g1 c2o2)
(and
(=>
(g2 c2o1)
(not
(f c2o2 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o2 c2o2)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(and
(=>
(g1 c2o1)
(or
(and
(g2 c2o1)
(not
(f c2o1 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o1 c2o2)
)
)
)
)
(=>
(g1 c2o2)
(or
(and
(g2 c2o1)
(not
(f c2o2 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o2 c2o2)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(and
(=>
(g1 c2o1)
(and
(=>
(g2 c2o1)
(not
(f c2o1 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o1 c2o2)
)
)
)
)
(=>
(g1 c2o2)
(and
(=>
(g2 c2o1)
(not
(f c2o2 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o2 c2o2)
)
)
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
(= q2 true)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
(g1 c3o3)
(g1 c3o4)
)
(and
(=>
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o1 c3o1)
)
(=>
(g2 c3o2)
(f c3o1 c3o2)
)
(=>
(g2 c3o3)
(f c3o1 c3o3)
)
(=>
(g2 c3o4)
(f c3o1 c3o4)
)
)
)
)
(=>
(g1 c3o2)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o2 c3o1)
)
(=>
(g2 c3o2)
(f c3o2 c3o2)
)
(=>
(g2 c3o3)
(f c3o2 c3o3)
)
(=>
(g2 c3o4)
(f c3o2 c3o4)
)
)
)
)
(=>
(g1 c3o3)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o3 c3o1)
)
(=>
(g2 c3o2)
(f c3o3 c3o2)
)
(=>
(g2 c3o3)
(f c3o3 c3o3)
)
(=>
(g2 c3o4)
(f c3o3 c3o4)
)
)
)
)
(=>
(g1 c3o4)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o4 c3o1)
)
(=>
(g2 c3o2)
(f c3o4 c3o2)
)
(=>
(g2 c3o3)
(f c3o4 c3o3)
)
(=>
(g2 c3o4)
(f c3o4 c3o4)
)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
(g1 c3o3)
(g1 c3o4)
)
(and
(=>
(g1 c3o1)
(or
(and
(g2 c3o1)
(f c3o1 c3o1)
)
(and
(g2 c3o2)
(f c3o1 c3o2)
)
(and
(g2 c3o3)
(f c3o1 c3o3)
)
(and
(g2 c3o4)
(f c3o1 c3o4)
)
)
)
(=>
(g1 c3o2)
(or
(and
(g2 c3o1)
(f c3o2 c3o1)
)
(and
(g2 c3o2)
(f c3o2 c3o2)
)
(and
(g2 c3o3)
(f c3o2 c3o3)
)
(and
(g2 c3o4)
(f c3o2 c3o4)
)
)
)
(=>
(g1 c3o3)
(or
(and
(g2 c3o1)
(f c3o3 c3o1)
)
(and
(g2 c3o2)
(f c3o3 c3o2)
)
(and
(g2 c3o3)
(f c3o3 c3o3)
)
(and
(g2 c3o4)
(f c3o3 c3o4)
)
)
)
(=>
(g1 c3o4)
(or
(and
(g2 c3o1)
(f c3o4 c3o1)
)
(and
(g2 c3o2)
(f c3o4 c3o2)
)
(and
(g2 c3o3)
(f c3o4 c3o3)
)
(and
(g2 c3o4)
(f c3o4 c3o4)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(or
(and
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o1 c3o1)
)
(=>
(g2 c3o2)
(f c3o1 c3o2)
)
(=>
(g2 c3o3)
(f c3o1 c3o3)
)
(=>
(g2 c3o4)
(f c3o1 c3o4)
)
)
)
)
(and
(g1 c3o2)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o2 c3o1)
)
(=>
(g2 c3o2)
(f c3o2 c3o2)
)
(=>
(g2 c3o3)
(f c3o2 c3o3)
)
(=>
(g2 c3o4)
(f c3o2 c3o4)
)
)
)
)
(and
(g1 c3o3)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o3 c3o1)
)
(=>
(g2 c3o2)
(f c3o3 c3o2)
)
(=>
(g2 c3o3)
(f c3o3 c3o3)
)
(=>
(g2 c3o4)
(f c3o3 c3o4)
)
)
)
)
(and
(g1 c3o4)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
)
(and
(=>
(g2 c3o1)
(f c3o4 c3o1)
)
(=>
(g2 c3o2)
(f c3o4 c3o2)
)
(=>
(g2 c3o3)
(f c3o4 c3o3)
)
(=>
(g2 c3o4)
(f c3o4 c3o4)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(or
(and
(g1 c3o1)
(or
(and
(g2 c3o1)
(f c3o1 c3o1)
)
(and
(g2 c3o2)
(f c3o1 c3o2)
)
(and
(g2 c3o3)
(f c3o1 c3o3)
)
(and
(g2 c3o4)
(f c3o1 c3o4)
)
)
)
(and
(g1 c3o2)
(or
(and
(g2 c3o1)
(f c3o2 c3o1)
)
(and
(g2 c3o2)
(f c3o2 c3o2)
)
(and
(g2 c3o3)
(f c3o2 c3o3)
)
(and
(g2 c3o4)
(f c3o2 c3o4)
)
)
)
(and
(g1 c3o3)
(or
(and
(g2 c3o1)
(f c3o3 c3o1)
)
(and
(g2 c3o2)
(f c3o3 c3o2)
)
(and
(g2 c3o3)
(f c3o3 c3o3)
)
(and
(g2 c3o4)
(f c3o3 c3o4)
)
)
)
(and
(g1 c3o4)
(or
(and
(g2 c3o1)
(f c3o4 c3o1)
)
(and
(g2 c3o2)
(f c3o4 c3o2)
)
(and
(g2 c3o3)
(f c3o4 c3o3)
)
(and
(g2 c3o4)
(f c3o4 c3o4)
)
)
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
(= q2 true)
)
(or
(and
(g1 c3o1)
(or
(and
(g2 c3o1)
(not
(f c3o1 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o1 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o1 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o1 c3o4)
)
)
)
)
(and
(g1 c3o2)
(or
(and
(g2 c3o1)
(not
(f c3o2 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o2 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o2 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o2 c3o4)
)
)
)
)
(and
(g1 c3o3)
(or
(and
(g2 c3o1)
(not
(f c3o3 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o3 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o3 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o3 c3o4)
)
)
)
)
(and
(g1 c3o4)
(or
(and
(g2 c3o1)
(not
(f c3o4 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o4 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o4 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o4 c3o4)
)
)
)
)
)
)
(=>
(and
(= q1 true)
(= q2 false)
)
(or
(and
(g1 c3o1)
(and
(=>
(g2 c3o1)
(not
(f c3o1 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o1 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o1 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o1 c3o4)
)
)
)
)
(and
(g1 c3o2)
(and
(=>
(g2 c3o1)
(not
(f c3o2 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o2 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o2 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o2 c3o4)
)
)
)
)
(and
(g1 c3o3)
(and
(=>
(g2 c3o1)
(not
(f c3o3 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o3 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o3 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o3 c3o4)
)
)
)
)
(and
(g1 c3o4)
(and
(=>
(g2 c3o1)
(not
(f c3o4 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o4 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o4 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o4 c3o4)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 true)
)
(and
(=>
(g1 c3o1)
(or
(and
(g2 c3o1)
(not
(f c3o1 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o1 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o1 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o1 c3o4)
)
)
)
)
(=>
(g1 c3o2)
(or
(and
(g2 c3o1)
(not
(f c3o2 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o2 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o2 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o2 c3o4)
)
)
)
)
(=>
(g1 c3o3)
(or
(and
(g2 c3o1)
(not
(f c3o3 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o3 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o3 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o3 c3o4)
)
)
)
)
(=>
(g1 c3o4)
(or
(and
(g2 c3o1)
(not
(f c3o4 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o4 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o4 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o4 c3o4)
)
)
)
)
)
)
(=>
(and
(= q1 false)
(= q2 false)
)
(and
(=>
(g1 c3o1)
(and
(=>
(g2 c3o1)
(not
(f c3o1 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o1 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o1 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o1 c3o4)
)
)
)
)
(=>
(g1 c3o2)
(and
(=>
(g2 c3o1)
(not
(f c3o2 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o2 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o2 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o2 c3o4)
)
)
)
)
(=>
(g1 c3o3)
(and
(=>
(g2 c3o1)
(not
(f c3o3 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o3 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o3 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o3 c3o4)
)
)
)
)
(=>
(g1 c3o4)
(and
(=>
(g2 c3o1)
(not
(f c3o4 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o4 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o4 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o4 c3o4)
)
)
)
)
)
)
)
)
)


(check-sat)
(get-model)