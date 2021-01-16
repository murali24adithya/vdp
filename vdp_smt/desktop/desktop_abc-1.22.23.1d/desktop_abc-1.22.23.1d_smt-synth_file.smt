(set-logic ALL)

(define-fun t1o1 () (_ BitVec 6) (_ bv0 6))
(define-fun t1o2 () (_ BitVec 6) (_ bv1 6))
(define-fun t1o3 () (_ BitVec 6) (_ bv2 6))
(define-fun t1o4 () (_ BitVec 6) (_ bv3 6))
(define-fun t1o5 () (_ BitVec 6) (_ bv4 6))
(define-fun t1o6 () (_ BitVec 6) (_ bv5 6))
(define-fun t1o7 () (_ BitVec 6) (_ bv6 6))
(define-fun t2o1 () (_ BitVec 6) (_ bv7 6))
(define-fun t2o2 () (_ BitVec 6) (_ bv8 6))
(define-fun t2o3 () (_ BitVec 6) (_ bv9 6))
(define-fun t2o4 () (_ BitVec 6) (_ bv10 6))
(define-fun t2o5 () (_ BitVec 6) (_ bv11 6))
(define-fun t2o6 () (_ BitVec 6) (_ bv12 6))
(define-fun t3o1 () (_ BitVec 6) (_ bv13 6))
(define-fun t3o2 () (_ BitVec 6) (_ bv14 6))
(define-fun t3o3 () (_ BitVec 6) (_ bv15 6))
(define-fun t3o4 () (_ BitVec 6) (_ bv16 6))
(define-fun c1o1 () (_ BitVec 6) (_ bv17 6))
(define-fun c1o2 () (_ BitVec 6) (_ bv18 6))
(define-fun c1o3 () (_ BitVec 6) (_ bv19 6))
(define-fun c1o4 () (_ BitVec 6) (_ bv20 6))
(define-fun c1o5 () (_ BitVec 6) (_ bv21 6))
(define-fun c1o6 () (_ BitVec 6) (_ bv22 6))
(define-fun c1o7 () (_ BitVec 6) (_ bv23 6))
(define-fun c1o8 () (_ BitVec 6) (_ bv24 6))
(define-fun c1o9 () (_ BitVec 6) (_ bv25 6))
(define-fun c1o10 () (_ BitVec 6) (_ bv26 6))
(define-fun c1o11 () (_ BitVec 6) (_ bv27 6))
(define-fun c1o12 () (_ BitVec 6) (_ bv28 6))
(define-fun c1o13 () (_ BitVec 6) (_ bv29 6))
(define-fun c1o14 () (_ BitVec 6) (_ bv30 6))
(define-fun c1o15 () (_ BitVec 6) (_ bv31 6))
(define-fun c1o16 () (_ BitVec 6) (_ bv32 6))
(define-fun c1o17 () (_ BitVec 6) (_ bv33 6))
(define-fun c1o18 () (_ BitVec 6) (_ bv34 6))
(define-fun c1o19 () (_ BitVec 6) (_ bv35 6))
(define-fun c2o1 () (_ BitVec 6) (_ bv36 6))
(define-fun c2o2 () (_ BitVec 6) (_ bv37 6))
(define-fun c2o3 () (_ BitVec 6) (_ bv38 6))
(define-fun c2o4 () (_ BitVec 6) (_ bv39 6))
(define-fun c3o1 () (_ BitVec 6) (_ bv40 6))
(define-fun c3o2 () (_ BitVec 6) (_ bv41 6))
(define-fun c3o3 () (_ BitVec 6) (_ bv42 6))
(define-fun c4o1 () (_ BitVec 6) (_ bv43 6))
(define-fun c4o2 () (_ BitVec 6) (_ bv44 6))
(define-fun c4o3 () (_ BitVec 6) (_ bv45 6))
(define-fun c4o4 () (_ BitVec 6) (_ bv46 6))
(define-fun c4o5 () (_ BitVec 6) (_ bv47 6))
(define-fun c4o6 () (_ BitVec 6) (_ bv48 6))
(define-fun c4o7 () (_ BitVec 6) (_ bv49 6))
(define-fun c4o8 () (_ BitVec 6) (_ bv50 6))
(define-fun c4o9 () (_ BitVec 6) (_ bv51 6))
(define-fun c4o10 () (_ BitVec 6) (_ bv52 6))
(define-fun c4o11 () (_ BitVec 6) (_ bv53 6))
(define-fun c4o12 () (_ BitVec 6) (_ bv54 6))


(define-fun vase () (_ BitVec 4) (_ bv0 4))
(define-fun mouse () (_ BitVec 4) (_ bv1 4))
(define-fun bed () (_ BitVec 4) (_ bv2 4))
(define-fun keyboard () (_ BitVec 4) (_ bv3 4))
(define-fun book () (_ BitVec 4) (_ bv4 4))
(define-fun tvmonitor () (_ BitVec 4) (_ bv5 4))
(define-fun diningtable () (_ BitVec 4) (_ bv6 4))
(define-fun person () (_ BitVec 4) (_ bv7 4))
(define-fun cell_phone () (_ BitVec 4) (_ bv8 4))
(define-fun chair () (_ BitVec 4) (_ bv9 4))
(define-fun bottle () (_ BitVec 4) (_ bv10 4))
(define-fun remote () (_ BitVec 4) (_ bv11 4))
(define-fun cup () (_ BitVec 4) (_ bv12 4))
(define-fun backpack () (_ BitVec 4) (_ bv13 4))


(define-fun labelOf((obj (_ BitVec 6))(lbl (_ BitVec 4))) Bool
(or
(and (= obj t1o1) (= lbl book))
(and (= obj t1o2) (= lbl cell_phone))
(and (= obj t1o3) (= lbl keyboard))
(and (= obj t1o4) (= lbl mouse))
(and (= obj t1o5) (= lbl tvmonitor))
(and (= obj t1o6) (= lbl cup))
(and (= obj t1o7) (= lbl bottle))
(and (= obj t2o1) (= lbl keyboard))
(and (= obj t2o2) (= lbl keyboard))
(and (= obj t2o3) (= lbl mouse))
(and (= obj t2o4) (= lbl mouse))
(and (= obj t2o5) (= lbl tvmonitor))
(and (= obj t2o6) (= lbl tvmonitor))
(and (= obj t3o1) (= lbl keyboard))
(and (= obj t3o2) (= lbl mouse))
(and (= obj t3o3) (= lbl tvmonitor))
(and (= obj t3o4) (= lbl bed))
(and (= obj c1o1) (= lbl tvmonitor))
(and (= obj c1o2) (= lbl chair))
(and (= obj c1o3) (= lbl chair))
(and (= obj c1o4) (= lbl chair))
(and (= obj c1o5) (= lbl cup))
(and (= obj c1o6) (= lbl cup))
(and (= obj c1o7) (= lbl cup))
(and (= obj c1o8) (= lbl cup))
(and (= obj c1o9) (= lbl bottle))
(and (= obj c1o10) (= lbl person))
(and (= obj c1o11) (= lbl person))
(and (= obj c1o12) (= lbl person))
(and (= obj c1o13) (= lbl person))
(and (= obj c1o14) (= lbl person))
(and (= obj c1o15) (= lbl person))
(and (= obj c1o16) (= lbl person))
(and (= obj c1o17) (= lbl person))
(and (= obj c1o18) (= lbl person))
(and (= obj c1o19) (= lbl person))
(and (= obj c2o1) (= lbl vase))
(and (= obj c2o2) (= lbl remote))
(and (= obj c2o3) (= lbl tvmonitor))
(and (= obj c2o4) (= lbl person))
(and (= obj c3o1) (= lbl keyboard))
(and (= obj c3o2) (= lbl mouse))
(and (= obj c3o3) (= lbl diningtable))
(and (= obj c4o1) (= lbl book))
(and (= obj c4o2) (= lbl book))
(and (= obj c4o3) (= lbl book))
(and (= obj c4o4) (= lbl keyboard))
(and (= obj c4o5) (= lbl mouse))
(and (= obj c4o6) (= lbl mouse))
(and (= obj c4o7) (= lbl mouse))
(and (= obj c4o8) (= lbl tvmonitor))
(and (= obj c4o9) (= lbl tvmonitor))
(and (= obj c4o10) (= lbl tvmonitor))
(and (= obj c4o11) (= lbl chair))
(and (= obj c4o12) (= lbl backpack))

))

(define-fun leftOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o3))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o7))
(and (= x t1o5) (= y t1o1))
(and (= x t1o5) (= y t1o4))
(and (= x t1o6) (= y t1o1))
(and (= x t1o6) (= y t1o3))
(and (= x t1o6) (= y t1o4))
(and (= x t1o6) (= y t1o5))
(and (= x t1o6) (= y t1o7))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o4))
(and (= x t2o1) (= y t2o4))
(and (= x t2o2) (= y t2o1))
(and (= x t2o2) (= y t2o4))
(and (= x t2o2) (= y t2o5))
(and (= x t2o3) (= y t2o1))
(and (= x t2o3) (= y t2o4))
(and (= x t2o3) (= y t2o5))
(and (= x t2o5) (= y t2o4))
(and (= x t2o6) (= y t2o1))
(and (= x t2o6) (= y t2o4))
(and (= x t2o6) (= y t2o5))
(and (= x t3o3) (= y t3o2))
(and (= x t3o3) (= y t3o4))
(and (= x c1o1) (= y c1o2))
(and (= x c1o1) (= y c1o3))
(and (= x c1o1) (= y c1o4))
(and (= x c1o1) (= y c1o5))
(and (= x c1o1) (= y c1o6))
(and (= x c1o1) (= y c1o7))
(and (= x c1o1) (= y c1o8))
(and (= x c1o1) (= y c1o9))
(and (= x c1o1) (= y c1o10))
(and (= x c1o1) (= y c1o11))
(and (= x c1o1) (= y c1o13))
(and (= x c1o1) (= y c1o14))
(and (= x c1o1) (= y c1o15))
(and (= x c1o1) (= y c1o16))
(and (= x c1o1) (= y c1o17))
(and (= x c1o1) (= y c1o18))
(and (= x c1o1) (= y c1o19))
(and (= x c1o2) (= y c1o4))
(and (= x c1o2) (= y c1o6))
(and (= x c1o2) (= y c1o8))
(and (= x c1o2) (= y c1o9))
(and (= x c1o2) (= y c1o10))
(and (= x c1o2) (= y c1o16))
(and (= x c1o3) (= y c1o4))
(and (= x c1o3) (= y c1o5))
(and (= x c1o3) (= y c1o6))
(and (= x c1o3) (= y c1o7))
(and (= x c1o3) (= y c1o8))
(and (= x c1o3) (= y c1o9))
(and (= x c1o3) (= y c1o10))
(and (= x c1o3) (= y c1o11))
(and (= x c1o3) (= y c1o15))
(and (= x c1o3) (= y c1o16))
(and (= x c1o5) (= y c1o4))
(and (= x c1o5) (= y c1o6))
(and (= x c1o5) (= y c1o8))
(and (= x c1o5) (= y c1o9))
(and (= x c1o5) (= y c1o10))
(and (= x c1o5) (= y c1o16))
(and (= x c1o7) (= y c1o4))
(and (= x c1o7) (= y c1o5))
(and (= x c1o7) (= y c1o6))
(and (= x c1o7) (= y c1o8))
(and (= x c1o7) (= y c1o9))
(and (= x c1o7) (= y c1o10))
(and (= x c1o7) (= y c1o15))
(and (= x c1o7) (= y c1o16))
(and (= x c1o8) (= y c1o4))
(and (= x c1o8) (= y c1o6))
(and (= x c1o9) (= y c1o4))
(and (= x c1o9) (= y c1o6))
(and (= x c1o11) (= y c1o4))
(and (= x c1o11) (= y c1o6))
(and (= x c1o11) (= y c1o8))
(and (= x c1o11) (= y c1o9))
(and (= x c1o11) (= y c1o10))
(and (= x c1o11) (= y c1o16))
(and (= x c1o12) (= y c1o2))
(and (= x c1o12) (= y c1o3))
(and (= x c1o12) (= y c1o4))
(and (= x c1o12) (= y c1o5))
(and (= x c1o12) (= y c1o6))
(and (= x c1o12) (= y c1o7))
(and (= x c1o12) (= y c1o8))
(and (= x c1o12) (= y c1o9))
(and (= x c1o12) (= y c1o10))
(and (= x c1o12) (= y c1o11))
(and (= x c1o12) (= y c1o13))
(and (= x c1o12) (= y c1o14))
(and (= x c1o12) (= y c1o15))
(and (= x c1o12) (= y c1o16))
(and (= x c1o12) (= y c1o17))
(and (= x c1o13) (= y c1o4))
(and (= x c1o13) (= y c1o6))
(and (= x c1o13) (= y c1o8))
(and (= x c1o13) (= y c1o9))
(and (= x c1o13) (= y c1o10))
(and (= x c1o13) (= y c1o16))
(and (= x c1o14) (= y c1o4))
(and (= x c1o14) (= y c1o5))
(and (= x c1o14) (= y c1o6))
(and (= x c1o14) (= y c1o7))
(and (= x c1o14) (= y c1o8))
(and (= x c1o14) (= y c1o9))
(and (= x c1o14) (= y c1o10))
(and (= x c1o14) (= y c1o11))
(and (= x c1o14) (= y c1o15))
(and (= x c1o14) (= y c1o16))
(and (= x c1o15) (= y c1o4))
(and (= x c1o15) (= y c1o6))
(and (= x c1o15) (= y c1o8))
(and (= x c1o15) (= y c1o9))
(and (= x c1o15) (= y c1o10))
(and (= x c1o17) (= y c1o2))
(and (= x c1o17) (= y c1o4))
(and (= x c1o17) (= y c1o5))
(and (= x c1o17) (= y c1o6))
(and (= x c1o17) (= y c1o7))
(and (= x c1o17) (= y c1o8))
(and (= x c1o17) (= y c1o9))
(and (= x c1o17) (= y c1o10))
(and (= x c1o17) (= y c1o11))
(and (= x c1o17) (= y c1o13))
(and (= x c1o17) (= y c1o15))
(and (= x c1o17) (= y c1o16))
(and (= x c1o18) (= y c1o2))
(and (= x c1o18) (= y c1o4))
(and (= x c1o18) (= y c1o5))
(and (= x c1o18) (= y c1o6))
(and (= x c1o18) (= y c1o7))
(and (= x c1o18) (= y c1o8))
(and (= x c1o18) (= y c1o9))
(and (= x c1o18) (= y c1o10))
(and (= x c1o18) (= y c1o11))
(and (= x c1o18) (= y c1o13))
(and (= x c1o18) (= y c1o15))
(and (= x c1o18) (= y c1o16))
(and (= x c1o19) (= y c1o2))
(and (= x c1o19) (= y c1o3))
(and (= x c1o19) (= y c1o4))
(and (= x c1o19) (= y c1o5))
(and (= x c1o19) (= y c1o6))
(and (= x c1o19) (= y c1o7))
(and (= x c1o19) (= y c1o8))
(and (= x c1o19) (= y c1o9))
(and (= x c1o19) (= y c1o10))
(and (= x c1o19) (= y c1o11))
(and (= x c1o19) (= y c1o13))
(and (= x c1o19) (= y c1o14))
(and (= x c1o19) (= y c1o15))
(and (= x c1o19) (= y c1o16))
(and (= x c1o19) (= y c1o17))
(and (= x c2o2) (= y c2o1))
(and (= x c2o2) (= y c2o3))
(and (= x c2o3) (= y c2o1))
(and (= x c2o4) (= y c2o1))
(and (= x c3o1) (= y c3o2))
(and (= x c4o1) (= y c4o4))
(and (= x c4o1) (= y c4o5))
(and (= x c4o1) (= y c4o6))
(and (= x c4o1) (= y c4o7))
(and (= x c4o1) (= y c4o8))
(and (= x c4o1) (= y c4o9))
(and (= x c4o1) (= y c4o10))
(and (= x c4o2) (= y c4o3))
(and (= x c4o2) (= y c4o4))
(and (= x c4o2) (= y c4o5))
(and (= x c4o2) (= y c4o6))
(and (= x c4o2) (= y c4o7))
(and (= x c4o2) (= y c4o8))
(and (= x c4o2) (= y c4o9))
(and (= x c4o2) (= y c4o10))
(and (= x c4o2) (= y c4o12))
(and (= x c4o3) (= y c4o4))
(and (= x c4o3) (= y c4o5))
(and (= x c4o3) (= y c4o6))
(and (= x c4o3) (= y c4o7))
(and (= x c4o3) (= y c4o8))
(and (= x c4o3) (= y c4o9))
(and (= x c4o3) (= y c4o10))
(and (= x c4o4) (= y c4o7))
(and (= x c4o5) (= y c4o7))
(and (= x c4o6) (= y c4o5))
(and (= x c4o6) (= y c4o7))
(and (= x c4o6) (= y c4o8))
(and (= x c4o9) (= y c4o5))
(and (= x c4o9) (= y c4o7))
(and (= x c4o10) (= y c4o4))
(and (= x c4o10) (= y c4o5))
(and (= x c4o10) (= y c4o6))
(and (= x c4o10) (= y c4o7))
(and (= x c4o10) (= y c4o8))
(and (= x c4o10) (= y c4o9))
(and (= x c4o11) (= y c4o4))
(and (= x c4o11) (= y c4o5))
(and (= x c4o11) (= y c4o7))
(and (= x c4o11) (= y c4o8))
(and (= x c4o12) (= y c4o4))
(and (= x c4o12) (= y c4o5))
(and (= x c4o12) (= y c4o6))
(and (= x c4o12) (= y c4o7))
(and (= x c4o12) (= y c4o8))
(and (= x c4o12) (= y c4o9))

))

(define-fun rightOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o5))
(and (= x t1o1) (= y t1o6))
(and (= x t1o1) (= y t1o7))
(and (= x t1o3) (= y t1o2))
(and (= x t1o3) (= y t1o6))
(and (= x t1o4) (= y t1o2))
(and (= x t1o4) (= y t1o5))
(and (= x t1o4) (= y t1o6))
(and (= x t1o4) (= y t1o7))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o6))
(and (= x t1o7) (= y t1o2))
(and (= x t1o7) (= y t1o6))
(and (= x t2o1) (= y t2o2))
(and (= x t2o1) (= y t2o3))
(and (= x t2o1) (= y t2o6))
(and (= x t2o4) (= y t2o1))
(and (= x t2o4) (= y t2o2))
(and (= x t2o4) (= y t2o3))
(and (= x t2o4) (= y t2o5))
(and (= x t2o4) (= y t2o6))
(and (= x t2o5) (= y t2o2))
(and (= x t2o5) (= y t2o3))
(and (= x t2o5) (= y t2o6))
(and (= x t3o2) (= y t3o3))
(and (= x t3o4) (= y t3o3))
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o12))
(and (= x c1o2) (= y c1o17))
(and (= x c1o2) (= y c1o18))
(and (= x c1o2) (= y c1o19))
(and (= x c1o3) (= y c1o1))
(and (= x c1o3) (= y c1o12))
(and (= x c1o3) (= y c1o19))
(and (= x c1o4) (= y c1o1))
(and (= x c1o4) (= y c1o2))
(and (= x c1o4) (= y c1o3))
(and (= x c1o4) (= y c1o5))
(and (= x c1o4) (= y c1o7))
(and (= x c1o4) (= y c1o8))
(and (= x c1o4) (= y c1o9))
(and (= x c1o4) (= y c1o11))
(and (= x c1o4) (= y c1o12))
(and (= x c1o4) (= y c1o13))
(and (= x c1o4) (= y c1o14))
(and (= x c1o4) (= y c1o15))
(and (= x c1o4) (= y c1o17))
(and (= x c1o4) (= y c1o18))
(and (= x c1o4) (= y c1o19))
(and (= x c1o5) (= y c1o1))
(and (= x c1o5) (= y c1o3))
(and (= x c1o5) (= y c1o7))
(and (= x c1o5) (= y c1o12))
(and (= x c1o5) (= y c1o14))
(and (= x c1o5) (= y c1o17))
(and (= x c1o5) (= y c1o18))
(and (= x c1o5) (= y c1o19))
(and (= x c1o6) (= y c1o1))
(and (= x c1o6) (= y c1o2))
(and (= x c1o6) (= y c1o3))
(and (= x c1o6) (= y c1o5))
(and (= x c1o6) (= y c1o7))
(and (= x c1o6) (= y c1o8))
(and (= x c1o6) (= y c1o9))
(and (= x c1o6) (= y c1o11))
(and (= x c1o6) (= y c1o12))
(and (= x c1o6) (= y c1o13))
(and (= x c1o6) (= y c1o14))
(and (= x c1o6) (= y c1o15))
(and (= x c1o6) (= y c1o17))
(and (= x c1o6) (= y c1o18))
(and (= x c1o6) (= y c1o19))
(and (= x c1o7) (= y c1o1))
(and (= x c1o7) (= y c1o3))
(and (= x c1o7) (= y c1o12))
(and (= x c1o7) (= y c1o14))
(and (= x c1o7) (= y c1o17))
(and (= x c1o7) (= y c1o18))
(and (= x c1o7) (= y c1o19))
(and (= x c1o8) (= y c1o1))
(and (= x c1o8) (= y c1o2))
(and (= x c1o8) (= y c1o3))
(and (= x c1o8) (= y c1o5))
(and (= x c1o8) (= y c1o7))
(and (= x c1o8) (= y c1o11))
(and (= x c1o8) (= y c1o12))
(and (= x c1o8) (= y c1o13))
(and (= x c1o8) (= y c1o14))
(and (= x c1o8) (= y c1o15))
(and (= x c1o8) (= y c1o17))
(and (= x c1o8) (= y c1o18))
(and (= x c1o8) (= y c1o19))
(and (= x c1o9) (= y c1o1))
(and (= x c1o9) (= y c1o2))
(and (= x c1o9) (= y c1o3))
(and (= x c1o9) (= y c1o5))
(and (= x c1o9) (= y c1o7))
(and (= x c1o9) (= y c1o11))
(and (= x c1o9) (= y c1o12))
(and (= x c1o9) (= y c1o13))
(and (= x c1o9) (= y c1o14))
(and (= x c1o9) (= y c1o15))
(and (= x c1o9) (= y c1o17))
(and (= x c1o9) (= y c1o18))
(and (= x c1o9) (= y c1o19))
(and (= x c1o10) (= y c1o1))
(and (= x c1o10) (= y c1o2))
(and (= x c1o10) (= y c1o3))
(and (= x c1o10) (= y c1o5))
(and (= x c1o10) (= y c1o7))
(and (= x c1o10) (= y c1o11))
(and (= x c1o10) (= y c1o12))
(and (= x c1o10) (= y c1o13))
(and (= x c1o10) (= y c1o14))
(and (= x c1o10) (= y c1o15))
(and (= x c1o10) (= y c1o17))
(and (= x c1o10) (= y c1o18))
(and (= x c1o10) (= y c1o19))
(and (= x c1o11) (= y c1o1))
(and (= x c1o11) (= y c1o3))
(and (= x c1o11) (= y c1o12))
(and (= x c1o11) (= y c1o14))
(and (= x c1o11) (= y c1o17))
(and (= x c1o11) (= y c1o18))
(and (= x c1o11) (= y c1o19))
(and (= x c1o13) (= y c1o1))
(and (= x c1o13) (= y c1o12))
(and (= x c1o13) (= y c1o17))
(and (= x c1o13) (= y c1o18))
(and (= x c1o13) (= y c1o19))
(and (= x c1o14) (= y c1o1))
(and (= x c1o14) (= y c1o12))
(and (= x c1o14) (= y c1o19))
(and (= x c1o15) (= y c1o1))
(and (= x c1o15) (= y c1o3))
(and (= x c1o15) (= y c1o7))
(and (= x c1o15) (= y c1o12))
(and (= x c1o15) (= y c1o14))
(and (= x c1o15) (= y c1o17))
(and (= x c1o15) (= y c1o18))
(and (= x c1o15) (= y c1o19))
(and (= x c1o16) (= y c1o1))
(and (= x c1o16) (= y c1o2))
(and (= x c1o16) (= y c1o3))
(and (= x c1o16) (= y c1o5))
(and (= x c1o16) (= y c1o7))
(and (= x c1o16) (= y c1o11))
(and (= x c1o16) (= y c1o12))
(and (= x c1o16) (= y c1o13))
(and (= x c1o16) (= y c1o14))
(and (= x c1o16) (= y c1o17))
(and (= x c1o16) (= y c1o18))
(and (= x c1o16) (= y c1o19))
(and (= x c1o17) (= y c1o1))
(and (= x c1o17) (= y c1o12))
(and (= x c1o17) (= y c1o19))
(and (= x c1o18) (= y c1o1))
(and (= x c1o19) (= y c1o1))
(and (= x c2o1) (= y c2o2))
(and (= x c2o1) (= y c2o3))
(and (= x c2o1) (= y c2o4))
(and (= x c2o3) (= y c2o2))
(and (= x c3o2) (= y c3o1))
(and (= x c4o3) (= y c4o2))
(and (= x c4o4) (= y c4o1))
(and (= x c4o4) (= y c4o2))
(and (= x c4o4) (= y c4o3))
(and (= x c4o4) (= y c4o10))
(and (= x c4o4) (= y c4o11))
(and (= x c4o4) (= y c4o12))
(and (= x c4o5) (= y c4o1))
(and (= x c4o5) (= y c4o2))
(and (= x c4o5) (= y c4o3))
(and (= x c4o5) (= y c4o6))
(and (= x c4o5) (= y c4o9))
(and (= x c4o5) (= y c4o10))
(and (= x c4o5) (= y c4o11))
(and (= x c4o5) (= y c4o12))
(and (= x c4o6) (= y c4o1))
(and (= x c4o6) (= y c4o2))
(and (= x c4o6) (= y c4o3))
(and (= x c4o6) (= y c4o10))
(and (= x c4o6) (= y c4o12))
(and (= x c4o7) (= y c4o1))
(and (= x c4o7) (= y c4o2))
(and (= x c4o7) (= y c4o3))
(and (= x c4o7) (= y c4o4))
(and (= x c4o7) (= y c4o5))
(and (= x c4o7) (= y c4o6))
(and (= x c4o7) (= y c4o9))
(and (= x c4o7) (= y c4o10))
(and (= x c4o7) (= y c4o11))
(and (= x c4o7) (= y c4o12))
(and (= x c4o8) (= y c4o1))
(and (= x c4o8) (= y c4o2))
(and (= x c4o8) (= y c4o3))
(and (= x c4o8) (= y c4o6))
(and (= x c4o8) (= y c4o10))
(and (= x c4o8) (= y c4o11))
(and (= x c4o8) (= y c4o12))
(and (= x c4o9) (= y c4o1))
(and (= x c4o9) (= y c4o2))
(and (= x c4o9) (= y c4o3))
(and (= x c4o9) (= y c4o10))
(and (= x c4o9) (= y c4o12))
(and (= x c4o10) (= y c4o1))
(and (= x c4o10) (= y c4o2))
(and (= x c4o10) (= y c4o3))
(and (= x c4o12) (= y c4o2))

))

(define-fun above ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o3))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o3))
(and (= x t1o5) (= y t1o4))
(and (= x t1o6) (= y t1o2))
(and (= x t1o7) (= y t1o2))
(and (= x t1o7) (= y t1o3))
(and (= x t2o5) (= y t2o1))
(and (= x t2o5) (= y t2o2))
(and (= x t2o5) (= y t2o3))
(and (= x t2o5) (= y t2o4))
(and (= x t2o6) (= y t2o1))
(and (= x t2o6) (= y t2o2))
(and (= x t2o6) (= y t2o3))
(and (= x t2o6) (= y t2o4))
(and (= x t3o2) (= y t3o4))
(and (= x t3o3) (= y t3o1))
(and (= x t3o3) (= y t3o4))
(and (= x c1o1) (= y c1o2))
(and (= x c1o1) (= y c1o3))
(and (= x c1o1) (= y c1o4))
(and (= x c1o1) (= y c1o5))
(and (= x c1o1) (= y c1o6))
(and (= x c1o1) (= y c1o7))
(and (= x c1o1) (= y c1o8))
(and (= x c1o1) (= y c1o9))
(and (= x c1o1) (= y c1o10))
(and (= x c1o1) (= y c1o11))
(and (= x c1o1) (= y c1o13))
(and (= x c1o1) (= y c1o14))
(and (= x c1o1) (= y c1o15))
(and (= x c1o1) (= y c1o16))
(and (= x c1o1) (= y c1o17))
(and (= x c1o1) (= y c1o18))
(and (= x c1o1) (= y c1o19))
(and (= x c1o5) (= y c1o4))
(and (= x c1o6) (= y c1o2))
(and (= x c1o6) (= y c1o4))
(and (= x c1o7) (= y c1o2))
(and (= x c1o7) (= y c1o3))
(and (= x c1o7) (= y c1o4))
(and (= x c1o7) (= y c1o5))
(and (= x c1o8) (= y c1o2))
(and (= x c1o8) (= y c1o3))
(and (= x c1o8) (= y c1o4))
(and (= x c1o8) (= y c1o5))
(and (= x c1o9) (= y c1o4))
(and (= x c1o11) (= y c1o2))
(and (= x c1o11) (= y c1o3))
(and (= x c1o11) (= y c1o4))
(and (= x c1o11) (= y c1o5))
(and (= x c1o11) (= y c1o6))
(and (= x c1o12) (= y c1o2))
(and (= x c1o12) (= y c1o3))
(and (= x c1o12) (= y c1o4))
(and (= x c1o15) (= y c1o2))
(and (= x c1o15) (= y c1o3))
(and (= x c1o15) (= y c1o4))
(and (= x c1o16) (= y c1o2))
(and (= x c1o16) (= y c1o3))
(and (= x c1o16) (= y c1o4))
(and (= x c1o17) (= y c1o2))
(and (= x c1o17) (= y c1o3))
(and (= x c1o17) (= y c1o4))
(and (= x c1o18) (= y c1o2))
(and (= x c1o18) (= y c1o3))
(and (= x c1o18) (= y c1o4))
(and (= x c1o18) (= y c1o5))
(and (= x c1o19) (= y c1o2))
(and (= x c1o19) (= y c1o3))
(and (= x c1o19) (= y c1o4))
(and (= x c1o19) (= y c1o5))
(and (= x c1o19) (= y c1o6))
(and (= x c1o19) (= y c1o7))
(and (= x c1o19) (= y c1o9))
(and (= x c2o1) (= y c2o3))
(and (= x c2o2) (= y c2o3))
(and (= x c4o1) (= y c4o4))
(and (= x c4o1) (= y c4o5))
(and (= x c4o1) (= y c4o6))
(and (= x c4o1) (= y c4o7))
(and (= x c4o1) (= y c4o8))
(and (= x c4o1) (= y c4o11))
(and (= x c4o1) (= y c4o12))
(and (= x c4o2) (= y c4o4))
(and (= x c4o2) (= y c4o5))
(and (= x c4o2) (= y c4o6))
(and (= x c4o2) (= y c4o7))
(and (= x c4o2) (= y c4o8))
(and (= x c4o2) (= y c4o11))
(and (= x c4o2) (= y c4o12))
(and (= x c4o3) (= y c4o4))
(and (= x c4o3) (= y c4o5))
(and (= x c4o3) (= y c4o6))
(and (= x c4o3) (= y c4o7))
(and (= x c4o3) (= y c4o8))
(and (= x c4o3) (= y c4o10))
(and (= x c4o3) (= y c4o11))
(and (= x c4o3) (= y c4o12))
(and (= x c4o4) (= y c4o5))
(and (= x c4o4) (= y c4o7))
(and (= x c4o4) (= y c4o12))
(and (= x c4o5) (= y c4o7))
(and (= x c4o6) (= y c4o5))
(and (= x c4o6) (= y c4o7))
(and (= x c4o6) (= y c4o11))
(and (= x c4o6) (= y c4o12))
(and (= x c4o8) (= y c4o5))
(and (= x c4o8) (= y c4o7))
(and (= x c4o8) (= y c4o11))
(and (= x c4o8) (= y c4o12))
(and (= x c4o9) (= y c4o4))
(and (= x c4o9) (= y c4o5))
(and (= x c4o9) (= y c4o6))
(and (= x c4o9) (= y c4o7))
(and (= x c4o9) (= y c4o11))
(and (= x c4o9) (= y c4o12))
(and (= x c4o10) (= y c4o4))
(and (= x c4o10) (= y c4o5))
(and (= x c4o10) (= y c4o6))
(and (= x c4o10) (= y c4o7))
(and (= x c4o10) (= y c4o11))
(and (= x c4o10) (= y c4o12))

))

(define-fun below ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o6))
(and (= x t1o2) (= y t1o7))
(and (= x t1o3) (= y t1o1))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o7))
(and (= x t1o4) (= y t1o5))
(and (= x t2o1) (= y t2o5))
(and (= x t2o1) (= y t2o6))
(and (= x t2o2) (= y t2o5))
(and (= x t2o2) (= y t2o6))
(and (= x t2o3) (= y t2o5))
(and (= x t2o3) (= y t2o6))
(and (= x t2o4) (= y t2o5))
(and (= x t2o4) (= y t2o6))
(and (= x t3o1) (= y t3o3))
(and (= x t3o4) (= y t3o2))
(and (= x t3o4) (= y t3o3))
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o6))
(and (= x c1o2) (= y c1o7))
(and (= x c1o2) (= y c1o8))
(and (= x c1o2) (= y c1o11))
(and (= x c1o2) (= y c1o12))
(and (= x c1o2) (= y c1o15))
(and (= x c1o2) (= y c1o16))
(and (= x c1o2) (= y c1o17))
(and (= x c1o2) (= y c1o18))
(and (= x c1o2) (= y c1o19))
(and (= x c1o3) (= y c1o1))
(and (= x c1o3) (= y c1o7))
(and (= x c1o3) (= y c1o8))
(and (= x c1o3) (= y c1o11))
(and (= x c1o3) (= y c1o12))
(and (= x c1o3) (= y c1o15))
(and (= x c1o3) (= y c1o16))
(and (= x c1o3) (= y c1o17))
(and (= x c1o3) (= y c1o18))
(and (= x c1o3) (= y c1o19))
(and (= x c1o4) (= y c1o1))
(and (= x c1o4) (= y c1o5))
(and (= x c1o4) (= y c1o6))
(and (= x c1o4) (= y c1o7))
(and (= x c1o4) (= y c1o8))
(and (= x c1o4) (= y c1o9))
(and (= x c1o4) (= y c1o11))
(and (= x c1o4) (= y c1o12))
(and (= x c1o4) (= y c1o15))
(and (= x c1o4) (= y c1o16))
(and (= x c1o4) (= y c1o17))
(and (= x c1o4) (= y c1o18))
(and (= x c1o4) (= y c1o19))
(and (= x c1o5) (= y c1o1))
(and (= x c1o5) (= y c1o7))
(and (= x c1o5) (= y c1o8))
(and (= x c1o5) (= y c1o11))
(and (= x c1o5) (= y c1o18))
(and (= x c1o5) (= y c1o19))
(and (= x c1o6) (= y c1o1))
(and (= x c1o6) (= y c1o11))
(and (= x c1o6) (= y c1o19))
(and (= x c1o7) (= y c1o1))
(and (= x c1o7) (= y c1o19))
(and (= x c1o8) (= y c1o1))
(and (= x c1o9) (= y c1o1))
(and (= x c1o9) (= y c1o19))
(and (= x c1o10) (= y c1o1))
(and (= x c1o11) (= y c1o1))
(and (= x c1o13) (= y c1o1))
(and (= x c1o14) (= y c1o1))
(and (= x c1o15) (= y c1o1))
(and (= x c1o16) (= y c1o1))
(and (= x c1o17) (= y c1o1))
(and (= x c1o18) (= y c1o1))
(and (= x c1o19) (= y c1o1))
(and (= x c2o3) (= y c2o1))
(and (= x c2o3) (= y c2o2))
(and (= x c4o4) (= y c4o1))
(and (= x c4o4) (= y c4o2))
(and (= x c4o4) (= y c4o3))
(and (= x c4o4) (= y c4o9))
(and (= x c4o4) (= y c4o10))
(and (= x c4o5) (= y c4o1))
(and (= x c4o5) (= y c4o2))
(and (= x c4o5) (= y c4o3))
(and (= x c4o5) (= y c4o4))
(and (= x c4o5) (= y c4o6))
(and (= x c4o5) (= y c4o8))
(and (= x c4o5) (= y c4o9))
(and (= x c4o5) (= y c4o10))
(and (= x c4o6) (= y c4o1))
(and (= x c4o6) (= y c4o2))
(and (= x c4o6) (= y c4o3))
(and (= x c4o6) (= y c4o9))
(and (= x c4o6) (= y c4o10))
(and (= x c4o7) (= y c4o1))
(and (= x c4o7) (= y c4o2))
(and (= x c4o7) (= y c4o3))
(and (= x c4o7) (= y c4o4))
(and (= x c4o7) (= y c4o5))
(and (= x c4o7) (= y c4o6))
(and (= x c4o7) (= y c4o8))
(and (= x c4o7) (= y c4o9))
(and (= x c4o7) (= y c4o10))
(and (= x c4o8) (= y c4o1))
(and (= x c4o8) (= y c4o2))
(and (= x c4o8) (= y c4o3))
(and (= x c4o10) (= y c4o3))
(and (= x c4o11) (= y c4o1))
(and (= x c4o11) (= y c4o2))
(and (= x c4o11) (= y c4o3))
(and (= x c4o11) (= y c4o6))
(and (= x c4o11) (= y c4o8))
(and (= x c4o11) (= y c4o9))
(and (= x c4o11) (= y c4o10))
(and (= x c4o12) (= y c4o1))
(and (= x c4o12) (= y c4o2))
(and (= x c4o12) (= y c4o3))
(and (= x c4o12) (= y c4o4))
(and (= x c4o12) (= y c4o6))
(and (= x c4o12) (= y c4o8))
(and (= x c4o12) (= y c4o9))
(and (= x c4o12) (= y c4o10))

))

(define-fun within ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x c1o5) (= y c1o13))
(and (= x c1o6) (= y c1o10))
(and (= x c1o7) (= y c1o13))
(and (= x c1o8) (= y c1o10))
(and (= x c1o8) (= y c1o16))
(and (= x c1o9) (= y c1o10))
(and (= x c3o2) (= y c3o3))
(and (= x c4o12) (= y c4o11))

))



(define-fun eq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (not (= x1 x2)))


(declare-const q1 Bool)
(declare-const q2 Bool)

(declare-const lbl_g1 (_ BitVec 4))
(assert (or
(= lbl_g1 vase)
(= lbl_g1 mouse)
(= lbl_g1 bed)
(= lbl_g1 keyboard)
(= lbl_g1 book)
(= lbl_g1 tvmonitor)
(= lbl_g1 diningtable)
(= lbl_g1 person)
(= lbl_g1 cell_phone)
(= lbl_g1 chair)
(= lbl_g1 bottle)
(= lbl_g1 remote)
(= lbl_g1 cup)
(= lbl_g1 backpack)
)
)
(define-fun g1 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g1)
)

(declare-const lbl_g2 (_ BitVec 4))
(assert (or
(= lbl_g2 vase)
(= lbl_g2 mouse)
(= lbl_g2 bed)
(= lbl_g2 keyboard)
(= lbl_g2 book)
(= lbl_g2 tvmonitor)
(= lbl_g2 diningtable)
(= lbl_g2 person)
(= lbl_g2 cell_phone)
(= lbl_g2 chair)
(= lbl_g2 bottle)
(= lbl_g2 remote)
(= lbl_g2 cup)
(= lbl_g2 backpack)
)
)
(define-fun g2 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g2)
)


(declare-const constval_vase (_ BitVec 4))
(assert (= constval_vase vase))
(declare-const constval_mouse (_ BitVec 4))
(assert (= constval_mouse mouse))
(declare-const constval_bed (_ BitVec 4))
(assert (= constval_bed bed))
(declare-const constval_keyboard (_ BitVec 4))
(assert (= constval_keyboard keyboard))
(declare-const constval_book (_ BitVec 4))
(assert (= constval_book book))
(declare-const constval_tvmonitor (_ BitVec 4))
(assert (= constval_tvmonitor tvmonitor))
(declare-const constval_diningtable (_ BitVec 4))
(assert (= constval_diningtable diningtable))
(declare-const constval_person (_ BitVec 4))
(assert (= constval_person person))
(declare-const constval_cell_phone (_ BitVec 4))
(assert (= constval_cell_phone cell_phone))
(declare-const constval_chair (_ BitVec 4))
(assert (= constval_chair chair))
(declare-const constval_bottle (_ BitVec 4))
(assert (= constval_bottle bottle))
(declare-const constval_remote (_ BitVec 4))
(assert (= constval_remote remote))
(declare-const constval_cup (_ BitVec 4))
(assert (= constval_cup cup))
(declare-const constval_backpack (_ BitVec 4))
(assert (= constval_backpack backpack))

(declare-const eq_x1_x1 Bool)
(declare-const eq_x1_x2 Bool)
(declare-const eq_x2_x1 Bool)
(declare-const eq_x2_x2 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const neq_x1_x2 Bool)
(declare-const neq_x2_x1 Bool)
(declare-const neq_x2_x2 Bool)
(declare-const labelOf_x1_vase Bool)
(declare-const labelOf_x1_mouse Bool)
(declare-const labelOf_x1_bed Bool)
(declare-const labelOf_x1_keyboard Bool)
(declare-const labelOf_x1_book Bool)
(declare-const labelOf_x1_tvmonitor Bool)
(declare-const labelOf_x1_diningtable Bool)
(declare-const labelOf_x1_person Bool)
(declare-const labelOf_x1_cell_phone Bool)
(declare-const labelOf_x1_chair Bool)
(declare-const labelOf_x1_bottle Bool)
(declare-const labelOf_x1_remote Bool)
(declare-const labelOf_x1_cup Bool)
(declare-const labelOf_x1_backpack Bool)
(declare-const labelOf_x2_vase Bool)
(declare-const labelOf_x2_mouse Bool)
(declare-const labelOf_x2_bed Bool)
(declare-const labelOf_x2_keyboard Bool)
(declare-const labelOf_x2_book Bool)
(declare-const labelOf_x2_tvmonitor Bool)
(declare-const labelOf_x2_diningtable Bool)
(declare-const labelOf_x2_person Bool)
(declare-const labelOf_x2_cell_phone Bool)
(declare-const labelOf_x2_chair Bool)
(declare-const labelOf_x2_bottle Bool)
(declare-const labelOf_x2_remote Bool)
(declare-const labelOf_x2_cup Bool)
(declare-const labelOf_x2_backpack Bool)
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
labelOf_x1_vase
(labelOf x1 vase)
)
(=>
labelOf_x1_mouse
(labelOf x1 mouse)
)
(=>
labelOf_x1_bed
(labelOf x1 bed)
)
(=>
labelOf_x1_keyboard
(labelOf x1 keyboard)
)
(=>
labelOf_x1_book
(labelOf x1 book)
)
(=>
labelOf_x1_tvmonitor
(labelOf x1 tvmonitor)
)
(=>
labelOf_x1_diningtable
(labelOf x1 diningtable)
)
(=>
labelOf_x1_person
(labelOf x1 person)
)
(=>
labelOf_x1_cell_phone
(labelOf x1 cell_phone)
)
(=>
labelOf_x1_chair
(labelOf x1 chair)
)
(=>
labelOf_x1_bottle
(labelOf x1 bottle)
)
(=>
labelOf_x1_remote
(labelOf x1 remote)
)
(=>
labelOf_x1_cup
(labelOf x1 cup)
)
(=>
labelOf_x1_backpack
(labelOf x1 backpack)
)
(=>
labelOf_x2_vase
(labelOf x2 vase)
)
(=>
labelOf_x2_mouse
(labelOf x2 mouse)
)
(=>
labelOf_x2_bed
(labelOf x2 bed)
)
(=>
labelOf_x2_keyboard
(labelOf x2 keyboard)
)
(=>
labelOf_x2_book
(labelOf x2 book)
)
(=>
labelOf_x2_tvmonitor
(labelOf x2 tvmonitor)
)
(=>
labelOf_x2_diningtable
(labelOf x2 diningtable)
)
(=>
labelOf_x2_person
(labelOf x2 person)
)
(=>
labelOf_x2_cell_phone
(labelOf x2 cell_phone)
)
(=>
labelOf_x2_chair
(labelOf x2 chair)
)
(=>
labelOf_x2_bottle
(labelOf x2 bottle)
)
(=>
labelOf_x2_remote
(labelOf x2 remote)
)
(=>
labelOf_x2_cup
(labelOf x2 cup)
)
(=>
labelOf_x2_backpack
(labelOf x2 backpack)
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
(declare-const c4 Bool)


(assert(or c1 c2 c3 c4))
(assert (=> c1 (and (not c2) (not c3) (not c4))))
(assert (=> c2 (and (not c1) (not c3) (not c4))))
(assert (=> c3 (and (not c1) (not c2) (not c4))))
(assert (=> c4 (and (not c1) (not c2) (not c3))))

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
(g1 t3o3)
(g1 t3o4)
)
(and
(=>
(g1 t3o1)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
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
(=>
(g2 t3o3)
(f t3o1 t3o3)
)
(=>
(g2 t3o4)
(f t3o1 t3o4)
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
(g2 t3o3)
(g2 t3o4)
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
(=>
(g2 t3o3)
(f t3o2 t3o3)
)
(=>
(g2 t3o4)
(f t3o2 t3o4)
)
)
)
)
(=>
(g1 t3o3)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
)
(and
(=>
(g2 t3o1)
(f t3o3 t3o1)
)
(=>
(g2 t3o2)
(f t3o3 t3o2)
)
(=>
(g2 t3o3)
(f t3o3 t3o3)
)
(=>
(g2 t3o4)
(f t3o3 t3o4)
)
)
)
)
(=>
(g1 t3o4)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
)
(and
(=>
(g2 t3o1)
(f t3o4 t3o1)
)
(=>
(g2 t3o2)
(f t3o4 t3o2)
)
(=>
(g2 t3o3)
(f t3o4 t3o3)
)
(=>
(g2 t3o4)
(f t3o4 t3o4)
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
(g1 t3o3)
(g1 t3o4)
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
(and
(g2 t3o3)
(f t3o1 t3o3)
)
(and
(g2 t3o4)
(f t3o1 t3o4)
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
(and
(g2 t3o3)
(f t3o2 t3o3)
)
(and
(g2 t3o4)
(f t3o2 t3o4)
)
)
)
(=>
(g1 t3o3)
(or
(and
(g2 t3o1)
(f t3o3 t3o1)
)
(and
(g2 t3o2)
(f t3o3 t3o2)
)
(and
(g2 t3o3)
(f t3o3 t3o3)
)
(and
(g2 t3o4)
(f t3o3 t3o4)
)
)
)
(=>
(g1 t3o4)
(or
(and
(g2 t3o1)
(f t3o4 t3o1)
)
(and
(g2 t3o2)
(f t3o4 t3o2)
)
(and
(g2 t3o3)
(f t3o4 t3o3)
)
(and
(g2 t3o4)
(f t3o4 t3o4)
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
(g2 t3o3)
(g2 t3o4)
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
(=>
(g2 t3o3)
(f t3o1 t3o3)
)
(=>
(g2 t3o4)
(f t3o1 t3o4)
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
(g2 t3o3)
(g2 t3o4)
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
(=>
(g2 t3o3)
(f t3o2 t3o3)
)
(=>
(g2 t3o4)
(f t3o2 t3o4)
)
)
)
)
(and
(g1 t3o3)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
)
(and
(=>
(g2 t3o1)
(f t3o3 t3o1)
)
(=>
(g2 t3o2)
(f t3o3 t3o2)
)
(=>
(g2 t3o3)
(f t3o3 t3o3)
)
(=>
(g2 t3o4)
(f t3o3 t3o4)
)
)
)
)
(and
(g1 t3o4)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
)
(and
(=>
(g2 t3o1)
(f t3o4 t3o1)
)
(=>
(g2 t3o2)
(f t3o4 t3o2)
)
(=>
(g2 t3o3)
(f t3o4 t3o3)
)
(=>
(g2 t3o4)
(f t3o4 t3o4)
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
(and
(g2 t3o3)
(f t3o1 t3o3)
)
(and
(g2 t3o4)
(f t3o1 t3o4)
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
(and
(g2 t3o3)
(f t3o2 t3o3)
)
(and
(g2 t3o4)
(f t3o2 t3o4)
)
)
)
(and
(g1 t3o3)
(or
(and
(g2 t3o1)
(f t3o3 t3o1)
)
(and
(g2 t3o2)
(f t3o3 t3o2)
)
(and
(g2 t3o3)
(f t3o3 t3o3)
)
(and
(g2 t3o4)
(f t3o3 t3o4)
)
)
)
(and
(g1 t3o4)
(or
(and
(g2 t3o1)
(f t3o4 t3o1)
)
(and
(g2 t3o2)
(f t3o4 t3o2)
)
(and
(g2 t3o3)
(f t3o4 t3o3)
)
(and
(g2 t3o4)
(f t3o4 t3o4)
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
(g1 c1o6)
(g1 c1o7)
(g1 c1o8)
(g1 c1o9)
(g1 c1o10)
(g1 c1o11)
(g1 c1o12)
(g1 c1o13)
(g1 c1o14)
(g1 c1o15)
(g1 c1o16)
(g1 c1o17)
(g1 c1o18)
(g1 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o1 c1o6)
)
(=>
(g2 c1o7)
(f c1o1 c1o7)
)
(=>
(g2 c1o8)
(f c1o1 c1o8)
)
(=>
(g2 c1o9)
(f c1o1 c1o9)
)
(=>
(g2 c1o10)
(f c1o1 c1o10)
)
(=>
(g2 c1o11)
(f c1o1 c1o11)
)
(=>
(g2 c1o12)
(f c1o1 c1o12)
)
(=>
(g2 c1o13)
(f c1o1 c1o13)
)
(=>
(g2 c1o14)
(f c1o1 c1o14)
)
(=>
(g2 c1o15)
(f c1o1 c1o15)
)
(=>
(g2 c1o16)
(f c1o1 c1o16)
)
(=>
(g2 c1o17)
(f c1o1 c1o17)
)
(=>
(g2 c1o18)
(f c1o1 c1o18)
)
(=>
(g2 c1o19)
(f c1o1 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o2 c1o6)
)
(=>
(g2 c1o7)
(f c1o2 c1o7)
)
(=>
(g2 c1o8)
(f c1o2 c1o8)
)
(=>
(g2 c1o9)
(f c1o2 c1o9)
)
(=>
(g2 c1o10)
(f c1o2 c1o10)
)
(=>
(g2 c1o11)
(f c1o2 c1o11)
)
(=>
(g2 c1o12)
(f c1o2 c1o12)
)
(=>
(g2 c1o13)
(f c1o2 c1o13)
)
(=>
(g2 c1o14)
(f c1o2 c1o14)
)
(=>
(g2 c1o15)
(f c1o2 c1o15)
)
(=>
(g2 c1o16)
(f c1o2 c1o16)
)
(=>
(g2 c1o17)
(f c1o2 c1o17)
)
(=>
(g2 c1o18)
(f c1o2 c1o18)
)
(=>
(g2 c1o19)
(f c1o2 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o3 c1o6)
)
(=>
(g2 c1o7)
(f c1o3 c1o7)
)
(=>
(g2 c1o8)
(f c1o3 c1o8)
)
(=>
(g2 c1o9)
(f c1o3 c1o9)
)
(=>
(g2 c1o10)
(f c1o3 c1o10)
)
(=>
(g2 c1o11)
(f c1o3 c1o11)
)
(=>
(g2 c1o12)
(f c1o3 c1o12)
)
(=>
(g2 c1o13)
(f c1o3 c1o13)
)
(=>
(g2 c1o14)
(f c1o3 c1o14)
)
(=>
(g2 c1o15)
(f c1o3 c1o15)
)
(=>
(g2 c1o16)
(f c1o3 c1o16)
)
(=>
(g2 c1o17)
(f c1o3 c1o17)
)
(=>
(g2 c1o18)
(f c1o3 c1o18)
)
(=>
(g2 c1o19)
(f c1o3 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o4 c1o6)
)
(=>
(g2 c1o7)
(f c1o4 c1o7)
)
(=>
(g2 c1o8)
(f c1o4 c1o8)
)
(=>
(g2 c1o9)
(f c1o4 c1o9)
)
(=>
(g2 c1o10)
(f c1o4 c1o10)
)
(=>
(g2 c1o11)
(f c1o4 c1o11)
)
(=>
(g2 c1o12)
(f c1o4 c1o12)
)
(=>
(g2 c1o13)
(f c1o4 c1o13)
)
(=>
(g2 c1o14)
(f c1o4 c1o14)
)
(=>
(g2 c1o15)
(f c1o4 c1o15)
)
(=>
(g2 c1o16)
(f c1o4 c1o16)
)
(=>
(g2 c1o17)
(f c1o4 c1o17)
)
(=>
(g2 c1o18)
(f c1o4 c1o18)
)
(=>
(g2 c1o19)
(f c1o4 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o5 c1o6)
)
(=>
(g2 c1o7)
(f c1o5 c1o7)
)
(=>
(g2 c1o8)
(f c1o5 c1o8)
)
(=>
(g2 c1o9)
(f c1o5 c1o9)
)
(=>
(g2 c1o10)
(f c1o5 c1o10)
)
(=>
(g2 c1o11)
(f c1o5 c1o11)
)
(=>
(g2 c1o12)
(f c1o5 c1o12)
)
(=>
(g2 c1o13)
(f c1o5 c1o13)
)
(=>
(g2 c1o14)
(f c1o5 c1o14)
)
(=>
(g2 c1o15)
(f c1o5 c1o15)
)
(=>
(g2 c1o16)
(f c1o5 c1o16)
)
(=>
(g2 c1o17)
(f c1o5 c1o17)
)
(=>
(g2 c1o18)
(f c1o5 c1o18)
)
(=>
(g2 c1o19)
(f c1o5 c1o19)
)
)
)
)
(=>
(g1 c1o6)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o6 c1o1)
)
(=>
(g2 c1o2)
(f c1o6 c1o2)
)
(=>
(g2 c1o3)
(f c1o6 c1o3)
)
(=>
(g2 c1o4)
(f c1o6 c1o4)
)
(=>
(g2 c1o5)
(f c1o6 c1o5)
)
(=>
(g2 c1o6)
(f c1o6 c1o6)
)
(=>
(g2 c1o7)
(f c1o6 c1o7)
)
(=>
(g2 c1o8)
(f c1o6 c1o8)
)
(=>
(g2 c1o9)
(f c1o6 c1o9)
)
(=>
(g2 c1o10)
(f c1o6 c1o10)
)
(=>
(g2 c1o11)
(f c1o6 c1o11)
)
(=>
(g2 c1o12)
(f c1o6 c1o12)
)
(=>
(g2 c1o13)
(f c1o6 c1o13)
)
(=>
(g2 c1o14)
(f c1o6 c1o14)
)
(=>
(g2 c1o15)
(f c1o6 c1o15)
)
(=>
(g2 c1o16)
(f c1o6 c1o16)
)
(=>
(g2 c1o17)
(f c1o6 c1o17)
)
(=>
(g2 c1o18)
(f c1o6 c1o18)
)
(=>
(g2 c1o19)
(f c1o6 c1o19)
)
)
)
)
(=>
(g1 c1o7)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o7 c1o1)
)
(=>
(g2 c1o2)
(f c1o7 c1o2)
)
(=>
(g2 c1o3)
(f c1o7 c1o3)
)
(=>
(g2 c1o4)
(f c1o7 c1o4)
)
(=>
(g2 c1o5)
(f c1o7 c1o5)
)
(=>
(g2 c1o6)
(f c1o7 c1o6)
)
(=>
(g2 c1o7)
(f c1o7 c1o7)
)
(=>
(g2 c1o8)
(f c1o7 c1o8)
)
(=>
(g2 c1o9)
(f c1o7 c1o9)
)
(=>
(g2 c1o10)
(f c1o7 c1o10)
)
(=>
(g2 c1o11)
(f c1o7 c1o11)
)
(=>
(g2 c1o12)
(f c1o7 c1o12)
)
(=>
(g2 c1o13)
(f c1o7 c1o13)
)
(=>
(g2 c1o14)
(f c1o7 c1o14)
)
(=>
(g2 c1o15)
(f c1o7 c1o15)
)
(=>
(g2 c1o16)
(f c1o7 c1o16)
)
(=>
(g2 c1o17)
(f c1o7 c1o17)
)
(=>
(g2 c1o18)
(f c1o7 c1o18)
)
(=>
(g2 c1o19)
(f c1o7 c1o19)
)
)
)
)
(=>
(g1 c1o8)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o8 c1o1)
)
(=>
(g2 c1o2)
(f c1o8 c1o2)
)
(=>
(g2 c1o3)
(f c1o8 c1o3)
)
(=>
(g2 c1o4)
(f c1o8 c1o4)
)
(=>
(g2 c1o5)
(f c1o8 c1o5)
)
(=>
(g2 c1o6)
(f c1o8 c1o6)
)
(=>
(g2 c1o7)
(f c1o8 c1o7)
)
(=>
(g2 c1o8)
(f c1o8 c1o8)
)
(=>
(g2 c1o9)
(f c1o8 c1o9)
)
(=>
(g2 c1o10)
(f c1o8 c1o10)
)
(=>
(g2 c1o11)
(f c1o8 c1o11)
)
(=>
(g2 c1o12)
(f c1o8 c1o12)
)
(=>
(g2 c1o13)
(f c1o8 c1o13)
)
(=>
(g2 c1o14)
(f c1o8 c1o14)
)
(=>
(g2 c1o15)
(f c1o8 c1o15)
)
(=>
(g2 c1o16)
(f c1o8 c1o16)
)
(=>
(g2 c1o17)
(f c1o8 c1o17)
)
(=>
(g2 c1o18)
(f c1o8 c1o18)
)
(=>
(g2 c1o19)
(f c1o8 c1o19)
)
)
)
)
(=>
(g1 c1o9)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o9 c1o1)
)
(=>
(g2 c1o2)
(f c1o9 c1o2)
)
(=>
(g2 c1o3)
(f c1o9 c1o3)
)
(=>
(g2 c1o4)
(f c1o9 c1o4)
)
(=>
(g2 c1o5)
(f c1o9 c1o5)
)
(=>
(g2 c1o6)
(f c1o9 c1o6)
)
(=>
(g2 c1o7)
(f c1o9 c1o7)
)
(=>
(g2 c1o8)
(f c1o9 c1o8)
)
(=>
(g2 c1o9)
(f c1o9 c1o9)
)
(=>
(g2 c1o10)
(f c1o9 c1o10)
)
(=>
(g2 c1o11)
(f c1o9 c1o11)
)
(=>
(g2 c1o12)
(f c1o9 c1o12)
)
(=>
(g2 c1o13)
(f c1o9 c1o13)
)
(=>
(g2 c1o14)
(f c1o9 c1o14)
)
(=>
(g2 c1o15)
(f c1o9 c1o15)
)
(=>
(g2 c1o16)
(f c1o9 c1o16)
)
(=>
(g2 c1o17)
(f c1o9 c1o17)
)
(=>
(g2 c1o18)
(f c1o9 c1o18)
)
(=>
(g2 c1o19)
(f c1o9 c1o19)
)
)
)
)
(=>
(g1 c1o10)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o10 c1o1)
)
(=>
(g2 c1o2)
(f c1o10 c1o2)
)
(=>
(g2 c1o3)
(f c1o10 c1o3)
)
(=>
(g2 c1o4)
(f c1o10 c1o4)
)
(=>
(g2 c1o5)
(f c1o10 c1o5)
)
(=>
(g2 c1o6)
(f c1o10 c1o6)
)
(=>
(g2 c1o7)
(f c1o10 c1o7)
)
(=>
(g2 c1o8)
(f c1o10 c1o8)
)
(=>
(g2 c1o9)
(f c1o10 c1o9)
)
(=>
(g2 c1o10)
(f c1o10 c1o10)
)
(=>
(g2 c1o11)
(f c1o10 c1o11)
)
(=>
(g2 c1o12)
(f c1o10 c1o12)
)
(=>
(g2 c1o13)
(f c1o10 c1o13)
)
(=>
(g2 c1o14)
(f c1o10 c1o14)
)
(=>
(g2 c1o15)
(f c1o10 c1o15)
)
(=>
(g2 c1o16)
(f c1o10 c1o16)
)
(=>
(g2 c1o17)
(f c1o10 c1o17)
)
(=>
(g2 c1o18)
(f c1o10 c1o18)
)
(=>
(g2 c1o19)
(f c1o10 c1o19)
)
)
)
)
(=>
(g1 c1o11)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o11 c1o1)
)
(=>
(g2 c1o2)
(f c1o11 c1o2)
)
(=>
(g2 c1o3)
(f c1o11 c1o3)
)
(=>
(g2 c1o4)
(f c1o11 c1o4)
)
(=>
(g2 c1o5)
(f c1o11 c1o5)
)
(=>
(g2 c1o6)
(f c1o11 c1o6)
)
(=>
(g2 c1o7)
(f c1o11 c1o7)
)
(=>
(g2 c1o8)
(f c1o11 c1o8)
)
(=>
(g2 c1o9)
(f c1o11 c1o9)
)
(=>
(g2 c1o10)
(f c1o11 c1o10)
)
(=>
(g2 c1o11)
(f c1o11 c1o11)
)
(=>
(g2 c1o12)
(f c1o11 c1o12)
)
(=>
(g2 c1o13)
(f c1o11 c1o13)
)
(=>
(g2 c1o14)
(f c1o11 c1o14)
)
(=>
(g2 c1o15)
(f c1o11 c1o15)
)
(=>
(g2 c1o16)
(f c1o11 c1o16)
)
(=>
(g2 c1o17)
(f c1o11 c1o17)
)
(=>
(g2 c1o18)
(f c1o11 c1o18)
)
(=>
(g2 c1o19)
(f c1o11 c1o19)
)
)
)
)
(=>
(g1 c1o12)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o12 c1o1)
)
(=>
(g2 c1o2)
(f c1o12 c1o2)
)
(=>
(g2 c1o3)
(f c1o12 c1o3)
)
(=>
(g2 c1o4)
(f c1o12 c1o4)
)
(=>
(g2 c1o5)
(f c1o12 c1o5)
)
(=>
(g2 c1o6)
(f c1o12 c1o6)
)
(=>
(g2 c1o7)
(f c1o12 c1o7)
)
(=>
(g2 c1o8)
(f c1o12 c1o8)
)
(=>
(g2 c1o9)
(f c1o12 c1o9)
)
(=>
(g2 c1o10)
(f c1o12 c1o10)
)
(=>
(g2 c1o11)
(f c1o12 c1o11)
)
(=>
(g2 c1o12)
(f c1o12 c1o12)
)
(=>
(g2 c1o13)
(f c1o12 c1o13)
)
(=>
(g2 c1o14)
(f c1o12 c1o14)
)
(=>
(g2 c1o15)
(f c1o12 c1o15)
)
(=>
(g2 c1o16)
(f c1o12 c1o16)
)
(=>
(g2 c1o17)
(f c1o12 c1o17)
)
(=>
(g2 c1o18)
(f c1o12 c1o18)
)
(=>
(g2 c1o19)
(f c1o12 c1o19)
)
)
)
)
(=>
(g1 c1o13)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o13 c1o1)
)
(=>
(g2 c1o2)
(f c1o13 c1o2)
)
(=>
(g2 c1o3)
(f c1o13 c1o3)
)
(=>
(g2 c1o4)
(f c1o13 c1o4)
)
(=>
(g2 c1o5)
(f c1o13 c1o5)
)
(=>
(g2 c1o6)
(f c1o13 c1o6)
)
(=>
(g2 c1o7)
(f c1o13 c1o7)
)
(=>
(g2 c1o8)
(f c1o13 c1o8)
)
(=>
(g2 c1o9)
(f c1o13 c1o9)
)
(=>
(g2 c1o10)
(f c1o13 c1o10)
)
(=>
(g2 c1o11)
(f c1o13 c1o11)
)
(=>
(g2 c1o12)
(f c1o13 c1o12)
)
(=>
(g2 c1o13)
(f c1o13 c1o13)
)
(=>
(g2 c1o14)
(f c1o13 c1o14)
)
(=>
(g2 c1o15)
(f c1o13 c1o15)
)
(=>
(g2 c1o16)
(f c1o13 c1o16)
)
(=>
(g2 c1o17)
(f c1o13 c1o17)
)
(=>
(g2 c1o18)
(f c1o13 c1o18)
)
(=>
(g2 c1o19)
(f c1o13 c1o19)
)
)
)
)
(=>
(g1 c1o14)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o14 c1o1)
)
(=>
(g2 c1o2)
(f c1o14 c1o2)
)
(=>
(g2 c1o3)
(f c1o14 c1o3)
)
(=>
(g2 c1o4)
(f c1o14 c1o4)
)
(=>
(g2 c1o5)
(f c1o14 c1o5)
)
(=>
(g2 c1o6)
(f c1o14 c1o6)
)
(=>
(g2 c1o7)
(f c1o14 c1o7)
)
(=>
(g2 c1o8)
(f c1o14 c1o8)
)
(=>
(g2 c1o9)
(f c1o14 c1o9)
)
(=>
(g2 c1o10)
(f c1o14 c1o10)
)
(=>
(g2 c1o11)
(f c1o14 c1o11)
)
(=>
(g2 c1o12)
(f c1o14 c1o12)
)
(=>
(g2 c1o13)
(f c1o14 c1o13)
)
(=>
(g2 c1o14)
(f c1o14 c1o14)
)
(=>
(g2 c1o15)
(f c1o14 c1o15)
)
(=>
(g2 c1o16)
(f c1o14 c1o16)
)
(=>
(g2 c1o17)
(f c1o14 c1o17)
)
(=>
(g2 c1o18)
(f c1o14 c1o18)
)
(=>
(g2 c1o19)
(f c1o14 c1o19)
)
)
)
)
(=>
(g1 c1o15)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o15 c1o1)
)
(=>
(g2 c1o2)
(f c1o15 c1o2)
)
(=>
(g2 c1o3)
(f c1o15 c1o3)
)
(=>
(g2 c1o4)
(f c1o15 c1o4)
)
(=>
(g2 c1o5)
(f c1o15 c1o5)
)
(=>
(g2 c1o6)
(f c1o15 c1o6)
)
(=>
(g2 c1o7)
(f c1o15 c1o7)
)
(=>
(g2 c1o8)
(f c1o15 c1o8)
)
(=>
(g2 c1o9)
(f c1o15 c1o9)
)
(=>
(g2 c1o10)
(f c1o15 c1o10)
)
(=>
(g2 c1o11)
(f c1o15 c1o11)
)
(=>
(g2 c1o12)
(f c1o15 c1o12)
)
(=>
(g2 c1o13)
(f c1o15 c1o13)
)
(=>
(g2 c1o14)
(f c1o15 c1o14)
)
(=>
(g2 c1o15)
(f c1o15 c1o15)
)
(=>
(g2 c1o16)
(f c1o15 c1o16)
)
(=>
(g2 c1o17)
(f c1o15 c1o17)
)
(=>
(g2 c1o18)
(f c1o15 c1o18)
)
(=>
(g2 c1o19)
(f c1o15 c1o19)
)
)
)
)
(=>
(g1 c1o16)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o16 c1o1)
)
(=>
(g2 c1o2)
(f c1o16 c1o2)
)
(=>
(g2 c1o3)
(f c1o16 c1o3)
)
(=>
(g2 c1o4)
(f c1o16 c1o4)
)
(=>
(g2 c1o5)
(f c1o16 c1o5)
)
(=>
(g2 c1o6)
(f c1o16 c1o6)
)
(=>
(g2 c1o7)
(f c1o16 c1o7)
)
(=>
(g2 c1o8)
(f c1o16 c1o8)
)
(=>
(g2 c1o9)
(f c1o16 c1o9)
)
(=>
(g2 c1o10)
(f c1o16 c1o10)
)
(=>
(g2 c1o11)
(f c1o16 c1o11)
)
(=>
(g2 c1o12)
(f c1o16 c1o12)
)
(=>
(g2 c1o13)
(f c1o16 c1o13)
)
(=>
(g2 c1o14)
(f c1o16 c1o14)
)
(=>
(g2 c1o15)
(f c1o16 c1o15)
)
(=>
(g2 c1o16)
(f c1o16 c1o16)
)
(=>
(g2 c1o17)
(f c1o16 c1o17)
)
(=>
(g2 c1o18)
(f c1o16 c1o18)
)
(=>
(g2 c1o19)
(f c1o16 c1o19)
)
)
)
)
(=>
(g1 c1o17)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o17 c1o1)
)
(=>
(g2 c1o2)
(f c1o17 c1o2)
)
(=>
(g2 c1o3)
(f c1o17 c1o3)
)
(=>
(g2 c1o4)
(f c1o17 c1o4)
)
(=>
(g2 c1o5)
(f c1o17 c1o5)
)
(=>
(g2 c1o6)
(f c1o17 c1o6)
)
(=>
(g2 c1o7)
(f c1o17 c1o7)
)
(=>
(g2 c1o8)
(f c1o17 c1o8)
)
(=>
(g2 c1o9)
(f c1o17 c1o9)
)
(=>
(g2 c1o10)
(f c1o17 c1o10)
)
(=>
(g2 c1o11)
(f c1o17 c1o11)
)
(=>
(g2 c1o12)
(f c1o17 c1o12)
)
(=>
(g2 c1o13)
(f c1o17 c1o13)
)
(=>
(g2 c1o14)
(f c1o17 c1o14)
)
(=>
(g2 c1o15)
(f c1o17 c1o15)
)
(=>
(g2 c1o16)
(f c1o17 c1o16)
)
(=>
(g2 c1o17)
(f c1o17 c1o17)
)
(=>
(g2 c1o18)
(f c1o17 c1o18)
)
(=>
(g2 c1o19)
(f c1o17 c1o19)
)
)
)
)
(=>
(g1 c1o18)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o18 c1o1)
)
(=>
(g2 c1o2)
(f c1o18 c1o2)
)
(=>
(g2 c1o3)
(f c1o18 c1o3)
)
(=>
(g2 c1o4)
(f c1o18 c1o4)
)
(=>
(g2 c1o5)
(f c1o18 c1o5)
)
(=>
(g2 c1o6)
(f c1o18 c1o6)
)
(=>
(g2 c1o7)
(f c1o18 c1o7)
)
(=>
(g2 c1o8)
(f c1o18 c1o8)
)
(=>
(g2 c1o9)
(f c1o18 c1o9)
)
(=>
(g2 c1o10)
(f c1o18 c1o10)
)
(=>
(g2 c1o11)
(f c1o18 c1o11)
)
(=>
(g2 c1o12)
(f c1o18 c1o12)
)
(=>
(g2 c1o13)
(f c1o18 c1o13)
)
(=>
(g2 c1o14)
(f c1o18 c1o14)
)
(=>
(g2 c1o15)
(f c1o18 c1o15)
)
(=>
(g2 c1o16)
(f c1o18 c1o16)
)
(=>
(g2 c1o17)
(f c1o18 c1o17)
)
(=>
(g2 c1o18)
(f c1o18 c1o18)
)
(=>
(g2 c1o19)
(f c1o18 c1o19)
)
)
)
)
(=>
(g1 c1o19)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o19 c1o1)
)
(=>
(g2 c1o2)
(f c1o19 c1o2)
)
(=>
(g2 c1o3)
(f c1o19 c1o3)
)
(=>
(g2 c1o4)
(f c1o19 c1o4)
)
(=>
(g2 c1o5)
(f c1o19 c1o5)
)
(=>
(g2 c1o6)
(f c1o19 c1o6)
)
(=>
(g2 c1o7)
(f c1o19 c1o7)
)
(=>
(g2 c1o8)
(f c1o19 c1o8)
)
(=>
(g2 c1o9)
(f c1o19 c1o9)
)
(=>
(g2 c1o10)
(f c1o19 c1o10)
)
(=>
(g2 c1o11)
(f c1o19 c1o11)
)
(=>
(g2 c1o12)
(f c1o19 c1o12)
)
(=>
(g2 c1o13)
(f c1o19 c1o13)
)
(=>
(g2 c1o14)
(f c1o19 c1o14)
)
(=>
(g2 c1o15)
(f c1o19 c1o15)
)
(=>
(g2 c1o16)
(f c1o19 c1o16)
)
(=>
(g2 c1o17)
(f c1o19 c1o17)
)
(=>
(g2 c1o18)
(f c1o19 c1o18)
)
(=>
(g2 c1o19)
(f c1o19 c1o19)
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
(g1 c1o6)
(g1 c1o7)
(g1 c1o8)
(g1 c1o9)
(g1 c1o10)
(g1 c1o11)
(g1 c1o12)
(g1 c1o13)
(g1 c1o14)
(g1 c1o15)
(g1 c1o16)
(g1 c1o17)
(g1 c1o18)
(g1 c1o19)
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
(and
(g2 c1o6)
(f c1o1 c1o6)
)
(and
(g2 c1o7)
(f c1o1 c1o7)
)
(and
(g2 c1o8)
(f c1o1 c1o8)
)
(and
(g2 c1o9)
(f c1o1 c1o9)
)
(and
(g2 c1o10)
(f c1o1 c1o10)
)
(and
(g2 c1o11)
(f c1o1 c1o11)
)
(and
(g2 c1o12)
(f c1o1 c1o12)
)
(and
(g2 c1o13)
(f c1o1 c1o13)
)
(and
(g2 c1o14)
(f c1o1 c1o14)
)
(and
(g2 c1o15)
(f c1o1 c1o15)
)
(and
(g2 c1o16)
(f c1o1 c1o16)
)
(and
(g2 c1o17)
(f c1o1 c1o17)
)
(and
(g2 c1o18)
(f c1o1 c1o18)
)
(and
(g2 c1o19)
(f c1o1 c1o19)
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
(and
(g2 c1o6)
(f c1o2 c1o6)
)
(and
(g2 c1o7)
(f c1o2 c1o7)
)
(and
(g2 c1o8)
(f c1o2 c1o8)
)
(and
(g2 c1o9)
(f c1o2 c1o9)
)
(and
(g2 c1o10)
(f c1o2 c1o10)
)
(and
(g2 c1o11)
(f c1o2 c1o11)
)
(and
(g2 c1o12)
(f c1o2 c1o12)
)
(and
(g2 c1o13)
(f c1o2 c1o13)
)
(and
(g2 c1o14)
(f c1o2 c1o14)
)
(and
(g2 c1o15)
(f c1o2 c1o15)
)
(and
(g2 c1o16)
(f c1o2 c1o16)
)
(and
(g2 c1o17)
(f c1o2 c1o17)
)
(and
(g2 c1o18)
(f c1o2 c1o18)
)
(and
(g2 c1o19)
(f c1o2 c1o19)
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
(and
(g2 c1o6)
(f c1o3 c1o6)
)
(and
(g2 c1o7)
(f c1o3 c1o7)
)
(and
(g2 c1o8)
(f c1o3 c1o8)
)
(and
(g2 c1o9)
(f c1o3 c1o9)
)
(and
(g2 c1o10)
(f c1o3 c1o10)
)
(and
(g2 c1o11)
(f c1o3 c1o11)
)
(and
(g2 c1o12)
(f c1o3 c1o12)
)
(and
(g2 c1o13)
(f c1o3 c1o13)
)
(and
(g2 c1o14)
(f c1o3 c1o14)
)
(and
(g2 c1o15)
(f c1o3 c1o15)
)
(and
(g2 c1o16)
(f c1o3 c1o16)
)
(and
(g2 c1o17)
(f c1o3 c1o17)
)
(and
(g2 c1o18)
(f c1o3 c1o18)
)
(and
(g2 c1o19)
(f c1o3 c1o19)
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
(and
(g2 c1o6)
(f c1o4 c1o6)
)
(and
(g2 c1o7)
(f c1o4 c1o7)
)
(and
(g2 c1o8)
(f c1o4 c1o8)
)
(and
(g2 c1o9)
(f c1o4 c1o9)
)
(and
(g2 c1o10)
(f c1o4 c1o10)
)
(and
(g2 c1o11)
(f c1o4 c1o11)
)
(and
(g2 c1o12)
(f c1o4 c1o12)
)
(and
(g2 c1o13)
(f c1o4 c1o13)
)
(and
(g2 c1o14)
(f c1o4 c1o14)
)
(and
(g2 c1o15)
(f c1o4 c1o15)
)
(and
(g2 c1o16)
(f c1o4 c1o16)
)
(and
(g2 c1o17)
(f c1o4 c1o17)
)
(and
(g2 c1o18)
(f c1o4 c1o18)
)
(and
(g2 c1o19)
(f c1o4 c1o19)
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
(and
(g2 c1o6)
(f c1o5 c1o6)
)
(and
(g2 c1o7)
(f c1o5 c1o7)
)
(and
(g2 c1o8)
(f c1o5 c1o8)
)
(and
(g2 c1o9)
(f c1o5 c1o9)
)
(and
(g2 c1o10)
(f c1o5 c1o10)
)
(and
(g2 c1o11)
(f c1o5 c1o11)
)
(and
(g2 c1o12)
(f c1o5 c1o12)
)
(and
(g2 c1o13)
(f c1o5 c1o13)
)
(and
(g2 c1o14)
(f c1o5 c1o14)
)
(and
(g2 c1o15)
(f c1o5 c1o15)
)
(and
(g2 c1o16)
(f c1o5 c1o16)
)
(and
(g2 c1o17)
(f c1o5 c1o17)
)
(and
(g2 c1o18)
(f c1o5 c1o18)
)
(and
(g2 c1o19)
(f c1o5 c1o19)
)
)
)
(=>
(g1 c1o6)
(or
(and
(g2 c1o1)
(f c1o6 c1o1)
)
(and
(g2 c1o2)
(f c1o6 c1o2)
)
(and
(g2 c1o3)
(f c1o6 c1o3)
)
(and
(g2 c1o4)
(f c1o6 c1o4)
)
(and
(g2 c1o5)
(f c1o6 c1o5)
)
(and
(g2 c1o6)
(f c1o6 c1o6)
)
(and
(g2 c1o7)
(f c1o6 c1o7)
)
(and
(g2 c1o8)
(f c1o6 c1o8)
)
(and
(g2 c1o9)
(f c1o6 c1o9)
)
(and
(g2 c1o10)
(f c1o6 c1o10)
)
(and
(g2 c1o11)
(f c1o6 c1o11)
)
(and
(g2 c1o12)
(f c1o6 c1o12)
)
(and
(g2 c1o13)
(f c1o6 c1o13)
)
(and
(g2 c1o14)
(f c1o6 c1o14)
)
(and
(g2 c1o15)
(f c1o6 c1o15)
)
(and
(g2 c1o16)
(f c1o6 c1o16)
)
(and
(g2 c1o17)
(f c1o6 c1o17)
)
(and
(g2 c1o18)
(f c1o6 c1o18)
)
(and
(g2 c1o19)
(f c1o6 c1o19)
)
)
)
(=>
(g1 c1o7)
(or
(and
(g2 c1o1)
(f c1o7 c1o1)
)
(and
(g2 c1o2)
(f c1o7 c1o2)
)
(and
(g2 c1o3)
(f c1o7 c1o3)
)
(and
(g2 c1o4)
(f c1o7 c1o4)
)
(and
(g2 c1o5)
(f c1o7 c1o5)
)
(and
(g2 c1o6)
(f c1o7 c1o6)
)
(and
(g2 c1o7)
(f c1o7 c1o7)
)
(and
(g2 c1o8)
(f c1o7 c1o8)
)
(and
(g2 c1o9)
(f c1o7 c1o9)
)
(and
(g2 c1o10)
(f c1o7 c1o10)
)
(and
(g2 c1o11)
(f c1o7 c1o11)
)
(and
(g2 c1o12)
(f c1o7 c1o12)
)
(and
(g2 c1o13)
(f c1o7 c1o13)
)
(and
(g2 c1o14)
(f c1o7 c1o14)
)
(and
(g2 c1o15)
(f c1o7 c1o15)
)
(and
(g2 c1o16)
(f c1o7 c1o16)
)
(and
(g2 c1o17)
(f c1o7 c1o17)
)
(and
(g2 c1o18)
(f c1o7 c1o18)
)
(and
(g2 c1o19)
(f c1o7 c1o19)
)
)
)
(=>
(g1 c1o8)
(or
(and
(g2 c1o1)
(f c1o8 c1o1)
)
(and
(g2 c1o2)
(f c1o8 c1o2)
)
(and
(g2 c1o3)
(f c1o8 c1o3)
)
(and
(g2 c1o4)
(f c1o8 c1o4)
)
(and
(g2 c1o5)
(f c1o8 c1o5)
)
(and
(g2 c1o6)
(f c1o8 c1o6)
)
(and
(g2 c1o7)
(f c1o8 c1o7)
)
(and
(g2 c1o8)
(f c1o8 c1o8)
)
(and
(g2 c1o9)
(f c1o8 c1o9)
)
(and
(g2 c1o10)
(f c1o8 c1o10)
)
(and
(g2 c1o11)
(f c1o8 c1o11)
)
(and
(g2 c1o12)
(f c1o8 c1o12)
)
(and
(g2 c1o13)
(f c1o8 c1o13)
)
(and
(g2 c1o14)
(f c1o8 c1o14)
)
(and
(g2 c1o15)
(f c1o8 c1o15)
)
(and
(g2 c1o16)
(f c1o8 c1o16)
)
(and
(g2 c1o17)
(f c1o8 c1o17)
)
(and
(g2 c1o18)
(f c1o8 c1o18)
)
(and
(g2 c1o19)
(f c1o8 c1o19)
)
)
)
(=>
(g1 c1o9)
(or
(and
(g2 c1o1)
(f c1o9 c1o1)
)
(and
(g2 c1o2)
(f c1o9 c1o2)
)
(and
(g2 c1o3)
(f c1o9 c1o3)
)
(and
(g2 c1o4)
(f c1o9 c1o4)
)
(and
(g2 c1o5)
(f c1o9 c1o5)
)
(and
(g2 c1o6)
(f c1o9 c1o6)
)
(and
(g2 c1o7)
(f c1o9 c1o7)
)
(and
(g2 c1o8)
(f c1o9 c1o8)
)
(and
(g2 c1o9)
(f c1o9 c1o9)
)
(and
(g2 c1o10)
(f c1o9 c1o10)
)
(and
(g2 c1o11)
(f c1o9 c1o11)
)
(and
(g2 c1o12)
(f c1o9 c1o12)
)
(and
(g2 c1o13)
(f c1o9 c1o13)
)
(and
(g2 c1o14)
(f c1o9 c1o14)
)
(and
(g2 c1o15)
(f c1o9 c1o15)
)
(and
(g2 c1o16)
(f c1o9 c1o16)
)
(and
(g2 c1o17)
(f c1o9 c1o17)
)
(and
(g2 c1o18)
(f c1o9 c1o18)
)
(and
(g2 c1o19)
(f c1o9 c1o19)
)
)
)
(=>
(g1 c1o10)
(or
(and
(g2 c1o1)
(f c1o10 c1o1)
)
(and
(g2 c1o2)
(f c1o10 c1o2)
)
(and
(g2 c1o3)
(f c1o10 c1o3)
)
(and
(g2 c1o4)
(f c1o10 c1o4)
)
(and
(g2 c1o5)
(f c1o10 c1o5)
)
(and
(g2 c1o6)
(f c1o10 c1o6)
)
(and
(g2 c1o7)
(f c1o10 c1o7)
)
(and
(g2 c1o8)
(f c1o10 c1o8)
)
(and
(g2 c1o9)
(f c1o10 c1o9)
)
(and
(g2 c1o10)
(f c1o10 c1o10)
)
(and
(g2 c1o11)
(f c1o10 c1o11)
)
(and
(g2 c1o12)
(f c1o10 c1o12)
)
(and
(g2 c1o13)
(f c1o10 c1o13)
)
(and
(g2 c1o14)
(f c1o10 c1o14)
)
(and
(g2 c1o15)
(f c1o10 c1o15)
)
(and
(g2 c1o16)
(f c1o10 c1o16)
)
(and
(g2 c1o17)
(f c1o10 c1o17)
)
(and
(g2 c1o18)
(f c1o10 c1o18)
)
(and
(g2 c1o19)
(f c1o10 c1o19)
)
)
)
(=>
(g1 c1o11)
(or
(and
(g2 c1o1)
(f c1o11 c1o1)
)
(and
(g2 c1o2)
(f c1o11 c1o2)
)
(and
(g2 c1o3)
(f c1o11 c1o3)
)
(and
(g2 c1o4)
(f c1o11 c1o4)
)
(and
(g2 c1o5)
(f c1o11 c1o5)
)
(and
(g2 c1o6)
(f c1o11 c1o6)
)
(and
(g2 c1o7)
(f c1o11 c1o7)
)
(and
(g2 c1o8)
(f c1o11 c1o8)
)
(and
(g2 c1o9)
(f c1o11 c1o9)
)
(and
(g2 c1o10)
(f c1o11 c1o10)
)
(and
(g2 c1o11)
(f c1o11 c1o11)
)
(and
(g2 c1o12)
(f c1o11 c1o12)
)
(and
(g2 c1o13)
(f c1o11 c1o13)
)
(and
(g2 c1o14)
(f c1o11 c1o14)
)
(and
(g2 c1o15)
(f c1o11 c1o15)
)
(and
(g2 c1o16)
(f c1o11 c1o16)
)
(and
(g2 c1o17)
(f c1o11 c1o17)
)
(and
(g2 c1o18)
(f c1o11 c1o18)
)
(and
(g2 c1o19)
(f c1o11 c1o19)
)
)
)
(=>
(g1 c1o12)
(or
(and
(g2 c1o1)
(f c1o12 c1o1)
)
(and
(g2 c1o2)
(f c1o12 c1o2)
)
(and
(g2 c1o3)
(f c1o12 c1o3)
)
(and
(g2 c1o4)
(f c1o12 c1o4)
)
(and
(g2 c1o5)
(f c1o12 c1o5)
)
(and
(g2 c1o6)
(f c1o12 c1o6)
)
(and
(g2 c1o7)
(f c1o12 c1o7)
)
(and
(g2 c1o8)
(f c1o12 c1o8)
)
(and
(g2 c1o9)
(f c1o12 c1o9)
)
(and
(g2 c1o10)
(f c1o12 c1o10)
)
(and
(g2 c1o11)
(f c1o12 c1o11)
)
(and
(g2 c1o12)
(f c1o12 c1o12)
)
(and
(g2 c1o13)
(f c1o12 c1o13)
)
(and
(g2 c1o14)
(f c1o12 c1o14)
)
(and
(g2 c1o15)
(f c1o12 c1o15)
)
(and
(g2 c1o16)
(f c1o12 c1o16)
)
(and
(g2 c1o17)
(f c1o12 c1o17)
)
(and
(g2 c1o18)
(f c1o12 c1o18)
)
(and
(g2 c1o19)
(f c1o12 c1o19)
)
)
)
(=>
(g1 c1o13)
(or
(and
(g2 c1o1)
(f c1o13 c1o1)
)
(and
(g2 c1o2)
(f c1o13 c1o2)
)
(and
(g2 c1o3)
(f c1o13 c1o3)
)
(and
(g2 c1o4)
(f c1o13 c1o4)
)
(and
(g2 c1o5)
(f c1o13 c1o5)
)
(and
(g2 c1o6)
(f c1o13 c1o6)
)
(and
(g2 c1o7)
(f c1o13 c1o7)
)
(and
(g2 c1o8)
(f c1o13 c1o8)
)
(and
(g2 c1o9)
(f c1o13 c1o9)
)
(and
(g2 c1o10)
(f c1o13 c1o10)
)
(and
(g2 c1o11)
(f c1o13 c1o11)
)
(and
(g2 c1o12)
(f c1o13 c1o12)
)
(and
(g2 c1o13)
(f c1o13 c1o13)
)
(and
(g2 c1o14)
(f c1o13 c1o14)
)
(and
(g2 c1o15)
(f c1o13 c1o15)
)
(and
(g2 c1o16)
(f c1o13 c1o16)
)
(and
(g2 c1o17)
(f c1o13 c1o17)
)
(and
(g2 c1o18)
(f c1o13 c1o18)
)
(and
(g2 c1o19)
(f c1o13 c1o19)
)
)
)
(=>
(g1 c1o14)
(or
(and
(g2 c1o1)
(f c1o14 c1o1)
)
(and
(g2 c1o2)
(f c1o14 c1o2)
)
(and
(g2 c1o3)
(f c1o14 c1o3)
)
(and
(g2 c1o4)
(f c1o14 c1o4)
)
(and
(g2 c1o5)
(f c1o14 c1o5)
)
(and
(g2 c1o6)
(f c1o14 c1o6)
)
(and
(g2 c1o7)
(f c1o14 c1o7)
)
(and
(g2 c1o8)
(f c1o14 c1o8)
)
(and
(g2 c1o9)
(f c1o14 c1o9)
)
(and
(g2 c1o10)
(f c1o14 c1o10)
)
(and
(g2 c1o11)
(f c1o14 c1o11)
)
(and
(g2 c1o12)
(f c1o14 c1o12)
)
(and
(g2 c1o13)
(f c1o14 c1o13)
)
(and
(g2 c1o14)
(f c1o14 c1o14)
)
(and
(g2 c1o15)
(f c1o14 c1o15)
)
(and
(g2 c1o16)
(f c1o14 c1o16)
)
(and
(g2 c1o17)
(f c1o14 c1o17)
)
(and
(g2 c1o18)
(f c1o14 c1o18)
)
(and
(g2 c1o19)
(f c1o14 c1o19)
)
)
)
(=>
(g1 c1o15)
(or
(and
(g2 c1o1)
(f c1o15 c1o1)
)
(and
(g2 c1o2)
(f c1o15 c1o2)
)
(and
(g2 c1o3)
(f c1o15 c1o3)
)
(and
(g2 c1o4)
(f c1o15 c1o4)
)
(and
(g2 c1o5)
(f c1o15 c1o5)
)
(and
(g2 c1o6)
(f c1o15 c1o6)
)
(and
(g2 c1o7)
(f c1o15 c1o7)
)
(and
(g2 c1o8)
(f c1o15 c1o8)
)
(and
(g2 c1o9)
(f c1o15 c1o9)
)
(and
(g2 c1o10)
(f c1o15 c1o10)
)
(and
(g2 c1o11)
(f c1o15 c1o11)
)
(and
(g2 c1o12)
(f c1o15 c1o12)
)
(and
(g2 c1o13)
(f c1o15 c1o13)
)
(and
(g2 c1o14)
(f c1o15 c1o14)
)
(and
(g2 c1o15)
(f c1o15 c1o15)
)
(and
(g2 c1o16)
(f c1o15 c1o16)
)
(and
(g2 c1o17)
(f c1o15 c1o17)
)
(and
(g2 c1o18)
(f c1o15 c1o18)
)
(and
(g2 c1o19)
(f c1o15 c1o19)
)
)
)
(=>
(g1 c1o16)
(or
(and
(g2 c1o1)
(f c1o16 c1o1)
)
(and
(g2 c1o2)
(f c1o16 c1o2)
)
(and
(g2 c1o3)
(f c1o16 c1o3)
)
(and
(g2 c1o4)
(f c1o16 c1o4)
)
(and
(g2 c1o5)
(f c1o16 c1o5)
)
(and
(g2 c1o6)
(f c1o16 c1o6)
)
(and
(g2 c1o7)
(f c1o16 c1o7)
)
(and
(g2 c1o8)
(f c1o16 c1o8)
)
(and
(g2 c1o9)
(f c1o16 c1o9)
)
(and
(g2 c1o10)
(f c1o16 c1o10)
)
(and
(g2 c1o11)
(f c1o16 c1o11)
)
(and
(g2 c1o12)
(f c1o16 c1o12)
)
(and
(g2 c1o13)
(f c1o16 c1o13)
)
(and
(g2 c1o14)
(f c1o16 c1o14)
)
(and
(g2 c1o15)
(f c1o16 c1o15)
)
(and
(g2 c1o16)
(f c1o16 c1o16)
)
(and
(g2 c1o17)
(f c1o16 c1o17)
)
(and
(g2 c1o18)
(f c1o16 c1o18)
)
(and
(g2 c1o19)
(f c1o16 c1o19)
)
)
)
(=>
(g1 c1o17)
(or
(and
(g2 c1o1)
(f c1o17 c1o1)
)
(and
(g2 c1o2)
(f c1o17 c1o2)
)
(and
(g2 c1o3)
(f c1o17 c1o3)
)
(and
(g2 c1o4)
(f c1o17 c1o4)
)
(and
(g2 c1o5)
(f c1o17 c1o5)
)
(and
(g2 c1o6)
(f c1o17 c1o6)
)
(and
(g2 c1o7)
(f c1o17 c1o7)
)
(and
(g2 c1o8)
(f c1o17 c1o8)
)
(and
(g2 c1o9)
(f c1o17 c1o9)
)
(and
(g2 c1o10)
(f c1o17 c1o10)
)
(and
(g2 c1o11)
(f c1o17 c1o11)
)
(and
(g2 c1o12)
(f c1o17 c1o12)
)
(and
(g2 c1o13)
(f c1o17 c1o13)
)
(and
(g2 c1o14)
(f c1o17 c1o14)
)
(and
(g2 c1o15)
(f c1o17 c1o15)
)
(and
(g2 c1o16)
(f c1o17 c1o16)
)
(and
(g2 c1o17)
(f c1o17 c1o17)
)
(and
(g2 c1o18)
(f c1o17 c1o18)
)
(and
(g2 c1o19)
(f c1o17 c1o19)
)
)
)
(=>
(g1 c1o18)
(or
(and
(g2 c1o1)
(f c1o18 c1o1)
)
(and
(g2 c1o2)
(f c1o18 c1o2)
)
(and
(g2 c1o3)
(f c1o18 c1o3)
)
(and
(g2 c1o4)
(f c1o18 c1o4)
)
(and
(g2 c1o5)
(f c1o18 c1o5)
)
(and
(g2 c1o6)
(f c1o18 c1o6)
)
(and
(g2 c1o7)
(f c1o18 c1o7)
)
(and
(g2 c1o8)
(f c1o18 c1o8)
)
(and
(g2 c1o9)
(f c1o18 c1o9)
)
(and
(g2 c1o10)
(f c1o18 c1o10)
)
(and
(g2 c1o11)
(f c1o18 c1o11)
)
(and
(g2 c1o12)
(f c1o18 c1o12)
)
(and
(g2 c1o13)
(f c1o18 c1o13)
)
(and
(g2 c1o14)
(f c1o18 c1o14)
)
(and
(g2 c1o15)
(f c1o18 c1o15)
)
(and
(g2 c1o16)
(f c1o18 c1o16)
)
(and
(g2 c1o17)
(f c1o18 c1o17)
)
(and
(g2 c1o18)
(f c1o18 c1o18)
)
(and
(g2 c1o19)
(f c1o18 c1o19)
)
)
)
(=>
(g1 c1o19)
(or
(and
(g2 c1o1)
(f c1o19 c1o1)
)
(and
(g2 c1o2)
(f c1o19 c1o2)
)
(and
(g2 c1o3)
(f c1o19 c1o3)
)
(and
(g2 c1o4)
(f c1o19 c1o4)
)
(and
(g2 c1o5)
(f c1o19 c1o5)
)
(and
(g2 c1o6)
(f c1o19 c1o6)
)
(and
(g2 c1o7)
(f c1o19 c1o7)
)
(and
(g2 c1o8)
(f c1o19 c1o8)
)
(and
(g2 c1o9)
(f c1o19 c1o9)
)
(and
(g2 c1o10)
(f c1o19 c1o10)
)
(and
(g2 c1o11)
(f c1o19 c1o11)
)
(and
(g2 c1o12)
(f c1o19 c1o12)
)
(and
(g2 c1o13)
(f c1o19 c1o13)
)
(and
(g2 c1o14)
(f c1o19 c1o14)
)
(and
(g2 c1o15)
(f c1o19 c1o15)
)
(and
(g2 c1o16)
(f c1o19 c1o16)
)
(and
(g2 c1o17)
(f c1o19 c1o17)
)
(and
(g2 c1o18)
(f c1o19 c1o18)
)
(and
(g2 c1o19)
(f c1o19 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o1 c1o6)
)
(=>
(g2 c1o7)
(f c1o1 c1o7)
)
(=>
(g2 c1o8)
(f c1o1 c1o8)
)
(=>
(g2 c1o9)
(f c1o1 c1o9)
)
(=>
(g2 c1o10)
(f c1o1 c1o10)
)
(=>
(g2 c1o11)
(f c1o1 c1o11)
)
(=>
(g2 c1o12)
(f c1o1 c1o12)
)
(=>
(g2 c1o13)
(f c1o1 c1o13)
)
(=>
(g2 c1o14)
(f c1o1 c1o14)
)
(=>
(g2 c1o15)
(f c1o1 c1o15)
)
(=>
(g2 c1o16)
(f c1o1 c1o16)
)
(=>
(g2 c1o17)
(f c1o1 c1o17)
)
(=>
(g2 c1o18)
(f c1o1 c1o18)
)
(=>
(g2 c1o19)
(f c1o1 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o2 c1o6)
)
(=>
(g2 c1o7)
(f c1o2 c1o7)
)
(=>
(g2 c1o8)
(f c1o2 c1o8)
)
(=>
(g2 c1o9)
(f c1o2 c1o9)
)
(=>
(g2 c1o10)
(f c1o2 c1o10)
)
(=>
(g2 c1o11)
(f c1o2 c1o11)
)
(=>
(g2 c1o12)
(f c1o2 c1o12)
)
(=>
(g2 c1o13)
(f c1o2 c1o13)
)
(=>
(g2 c1o14)
(f c1o2 c1o14)
)
(=>
(g2 c1o15)
(f c1o2 c1o15)
)
(=>
(g2 c1o16)
(f c1o2 c1o16)
)
(=>
(g2 c1o17)
(f c1o2 c1o17)
)
(=>
(g2 c1o18)
(f c1o2 c1o18)
)
(=>
(g2 c1o19)
(f c1o2 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o3 c1o6)
)
(=>
(g2 c1o7)
(f c1o3 c1o7)
)
(=>
(g2 c1o8)
(f c1o3 c1o8)
)
(=>
(g2 c1o9)
(f c1o3 c1o9)
)
(=>
(g2 c1o10)
(f c1o3 c1o10)
)
(=>
(g2 c1o11)
(f c1o3 c1o11)
)
(=>
(g2 c1o12)
(f c1o3 c1o12)
)
(=>
(g2 c1o13)
(f c1o3 c1o13)
)
(=>
(g2 c1o14)
(f c1o3 c1o14)
)
(=>
(g2 c1o15)
(f c1o3 c1o15)
)
(=>
(g2 c1o16)
(f c1o3 c1o16)
)
(=>
(g2 c1o17)
(f c1o3 c1o17)
)
(=>
(g2 c1o18)
(f c1o3 c1o18)
)
(=>
(g2 c1o19)
(f c1o3 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o4 c1o6)
)
(=>
(g2 c1o7)
(f c1o4 c1o7)
)
(=>
(g2 c1o8)
(f c1o4 c1o8)
)
(=>
(g2 c1o9)
(f c1o4 c1o9)
)
(=>
(g2 c1o10)
(f c1o4 c1o10)
)
(=>
(g2 c1o11)
(f c1o4 c1o11)
)
(=>
(g2 c1o12)
(f c1o4 c1o12)
)
(=>
(g2 c1o13)
(f c1o4 c1o13)
)
(=>
(g2 c1o14)
(f c1o4 c1o14)
)
(=>
(g2 c1o15)
(f c1o4 c1o15)
)
(=>
(g2 c1o16)
(f c1o4 c1o16)
)
(=>
(g2 c1o17)
(f c1o4 c1o17)
)
(=>
(g2 c1o18)
(f c1o4 c1o18)
)
(=>
(g2 c1o19)
(f c1o4 c1o19)
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
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
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
(=>
(g2 c1o6)
(f c1o5 c1o6)
)
(=>
(g2 c1o7)
(f c1o5 c1o7)
)
(=>
(g2 c1o8)
(f c1o5 c1o8)
)
(=>
(g2 c1o9)
(f c1o5 c1o9)
)
(=>
(g2 c1o10)
(f c1o5 c1o10)
)
(=>
(g2 c1o11)
(f c1o5 c1o11)
)
(=>
(g2 c1o12)
(f c1o5 c1o12)
)
(=>
(g2 c1o13)
(f c1o5 c1o13)
)
(=>
(g2 c1o14)
(f c1o5 c1o14)
)
(=>
(g2 c1o15)
(f c1o5 c1o15)
)
(=>
(g2 c1o16)
(f c1o5 c1o16)
)
(=>
(g2 c1o17)
(f c1o5 c1o17)
)
(=>
(g2 c1o18)
(f c1o5 c1o18)
)
(=>
(g2 c1o19)
(f c1o5 c1o19)
)
)
)
)
(and
(g1 c1o6)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o6 c1o1)
)
(=>
(g2 c1o2)
(f c1o6 c1o2)
)
(=>
(g2 c1o3)
(f c1o6 c1o3)
)
(=>
(g2 c1o4)
(f c1o6 c1o4)
)
(=>
(g2 c1o5)
(f c1o6 c1o5)
)
(=>
(g2 c1o6)
(f c1o6 c1o6)
)
(=>
(g2 c1o7)
(f c1o6 c1o7)
)
(=>
(g2 c1o8)
(f c1o6 c1o8)
)
(=>
(g2 c1o9)
(f c1o6 c1o9)
)
(=>
(g2 c1o10)
(f c1o6 c1o10)
)
(=>
(g2 c1o11)
(f c1o6 c1o11)
)
(=>
(g2 c1o12)
(f c1o6 c1o12)
)
(=>
(g2 c1o13)
(f c1o6 c1o13)
)
(=>
(g2 c1o14)
(f c1o6 c1o14)
)
(=>
(g2 c1o15)
(f c1o6 c1o15)
)
(=>
(g2 c1o16)
(f c1o6 c1o16)
)
(=>
(g2 c1o17)
(f c1o6 c1o17)
)
(=>
(g2 c1o18)
(f c1o6 c1o18)
)
(=>
(g2 c1o19)
(f c1o6 c1o19)
)
)
)
)
(and
(g1 c1o7)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o7 c1o1)
)
(=>
(g2 c1o2)
(f c1o7 c1o2)
)
(=>
(g2 c1o3)
(f c1o7 c1o3)
)
(=>
(g2 c1o4)
(f c1o7 c1o4)
)
(=>
(g2 c1o5)
(f c1o7 c1o5)
)
(=>
(g2 c1o6)
(f c1o7 c1o6)
)
(=>
(g2 c1o7)
(f c1o7 c1o7)
)
(=>
(g2 c1o8)
(f c1o7 c1o8)
)
(=>
(g2 c1o9)
(f c1o7 c1o9)
)
(=>
(g2 c1o10)
(f c1o7 c1o10)
)
(=>
(g2 c1o11)
(f c1o7 c1o11)
)
(=>
(g2 c1o12)
(f c1o7 c1o12)
)
(=>
(g2 c1o13)
(f c1o7 c1o13)
)
(=>
(g2 c1o14)
(f c1o7 c1o14)
)
(=>
(g2 c1o15)
(f c1o7 c1o15)
)
(=>
(g2 c1o16)
(f c1o7 c1o16)
)
(=>
(g2 c1o17)
(f c1o7 c1o17)
)
(=>
(g2 c1o18)
(f c1o7 c1o18)
)
(=>
(g2 c1o19)
(f c1o7 c1o19)
)
)
)
)
(and
(g1 c1o8)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o8 c1o1)
)
(=>
(g2 c1o2)
(f c1o8 c1o2)
)
(=>
(g2 c1o3)
(f c1o8 c1o3)
)
(=>
(g2 c1o4)
(f c1o8 c1o4)
)
(=>
(g2 c1o5)
(f c1o8 c1o5)
)
(=>
(g2 c1o6)
(f c1o8 c1o6)
)
(=>
(g2 c1o7)
(f c1o8 c1o7)
)
(=>
(g2 c1o8)
(f c1o8 c1o8)
)
(=>
(g2 c1o9)
(f c1o8 c1o9)
)
(=>
(g2 c1o10)
(f c1o8 c1o10)
)
(=>
(g2 c1o11)
(f c1o8 c1o11)
)
(=>
(g2 c1o12)
(f c1o8 c1o12)
)
(=>
(g2 c1o13)
(f c1o8 c1o13)
)
(=>
(g2 c1o14)
(f c1o8 c1o14)
)
(=>
(g2 c1o15)
(f c1o8 c1o15)
)
(=>
(g2 c1o16)
(f c1o8 c1o16)
)
(=>
(g2 c1o17)
(f c1o8 c1o17)
)
(=>
(g2 c1o18)
(f c1o8 c1o18)
)
(=>
(g2 c1o19)
(f c1o8 c1o19)
)
)
)
)
(and
(g1 c1o9)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o9 c1o1)
)
(=>
(g2 c1o2)
(f c1o9 c1o2)
)
(=>
(g2 c1o3)
(f c1o9 c1o3)
)
(=>
(g2 c1o4)
(f c1o9 c1o4)
)
(=>
(g2 c1o5)
(f c1o9 c1o5)
)
(=>
(g2 c1o6)
(f c1o9 c1o6)
)
(=>
(g2 c1o7)
(f c1o9 c1o7)
)
(=>
(g2 c1o8)
(f c1o9 c1o8)
)
(=>
(g2 c1o9)
(f c1o9 c1o9)
)
(=>
(g2 c1o10)
(f c1o9 c1o10)
)
(=>
(g2 c1o11)
(f c1o9 c1o11)
)
(=>
(g2 c1o12)
(f c1o9 c1o12)
)
(=>
(g2 c1o13)
(f c1o9 c1o13)
)
(=>
(g2 c1o14)
(f c1o9 c1o14)
)
(=>
(g2 c1o15)
(f c1o9 c1o15)
)
(=>
(g2 c1o16)
(f c1o9 c1o16)
)
(=>
(g2 c1o17)
(f c1o9 c1o17)
)
(=>
(g2 c1o18)
(f c1o9 c1o18)
)
(=>
(g2 c1o19)
(f c1o9 c1o19)
)
)
)
)
(and
(g1 c1o10)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o10 c1o1)
)
(=>
(g2 c1o2)
(f c1o10 c1o2)
)
(=>
(g2 c1o3)
(f c1o10 c1o3)
)
(=>
(g2 c1o4)
(f c1o10 c1o4)
)
(=>
(g2 c1o5)
(f c1o10 c1o5)
)
(=>
(g2 c1o6)
(f c1o10 c1o6)
)
(=>
(g2 c1o7)
(f c1o10 c1o7)
)
(=>
(g2 c1o8)
(f c1o10 c1o8)
)
(=>
(g2 c1o9)
(f c1o10 c1o9)
)
(=>
(g2 c1o10)
(f c1o10 c1o10)
)
(=>
(g2 c1o11)
(f c1o10 c1o11)
)
(=>
(g2 c1o12)
(f c1o10 c1o12)
)
(=>
(g2 c1o13)
(f c1o10 c1o13)
)
(=>
(g2 c1o14)
(f c1o10 c1o14)
)
(=>
(g2 c1o15)
(f c1o10 c1o15)
)
(=>
(g2 c1o16)
(f c1o10 c1o16)
)
(=>
(g2 c1o17)
(f c1o10 c1o17)
)
(=>
(g2 c1o18)
(f c1o10 c1o18)
)
(=>
(g2 c1o19)
(f c1o10 c1o19)
)
)
)
)
(and
(g1 c1o11)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o11 c1o1)
)
(=>
(g2 c1o2)
(f c1o11 c1o2)
)
(=>
(g2 c1o3)
(f c1o11 c1o3)
)
(=>
(g2 c1o4)
(f c1o11 c1o4)
)
(=>
(g2 c1o5)
(f c1o11 c1o5)
)
(=>
(g2 c1o6)
(f c1o11 c1o6)
)
(=>
(g2 c1o7)
(f c1o11 c1o7)
)
(=>
(g2 c1o8)
(f c1o11 c1o8)
)
(=>
(g2 c1o9)
(f c1o11 c1o9)
)
(=>
(g2 c1o10)
(f c1o11 c1o10)
)
(=>
(g2 c1o11)
(f c1o11 c1o11)
)
(=>
(g2 c1o12)
(f c1o11 c1o12)
)
(=>
(g2 c1o13)
(f c1o11 c1o13)
)
(=>
(g2 c1o14)
(f c1o11 c1o14)
)
(=>
(g2 c1o15)
(f c1o11 c1o15)
)
(=>
(g2 c1o16)
(f c1o11 c1o16)
)
(=>
(g2 c1o17)
(f c1o11 c1o17)
)
(=>
(g2 c1o18)
(f c1o11 c1o18)
)
(=>
(g2 c1o19)
(f c1o11 c1o19)
)
)
)
)
(and
(g1 c1o12)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o12 c1o1)
)
(=>
(g2 c1o2)
(f c1o12 c1o2)
)
(=>
(g2 c1o3)
(f c1o12 c1o3)
)
(=>
(g2 c1o4)
(f c1o12 c1o4)
)
(=>
(g2 c1o5)
(f c1o12 c1o5)
)
(=>
(g2 c1o6)
(f c1o12 c1o6)
)
(=>
(g2 c1o7)
(f c1o12 c1o7)
)
(=>
(g2 c1o8)
(f c1o12 c1o8)
)
(=>
(g2 c1o9)
(f c1o12 c1o9)
)
(=>
(g2 c1o10)
(f c1o12 c1o10)
)
(=>
(g2 c1o11)
(f c1o12 c1o11)
)
(=>
(g2 c1o12)
(f c1o12 c1o12)
)
(=>
(g2 c1o13)
(f c1o12 c1o13)
)
(=>
(g2 c1o14)
(f c1o12 c1o14)
)
(=>
(g2 c1o15)
(f c1o12 c1o15)
)
(=>
(g2 c1o16)
(f c1o12 c1o16)
)
(=>
(g2 c1o17)
(f c1o12 c1o17)
)
(=>
(g2 c1o18)
(f c1o12 c1o18)
)
(=>
(g2 c1o19)
(f c1o12 c1o19)
)
)
)
)
(and
(g1 c1o13)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o13 c1o1)
)
(=>
(g2 c1o2)
(f c1o13 c1o2)
)
(=>
(g2 c1o3)
(f c1o13 c1o3)
)
(=>
(g2 c1o4)
(f c1o13 c1o4)
)
(=>
(g2 c1o5)
(f c1o13 c1o5)
)
(=>
(g2 c1o6)
(f c1o13 c1o6)
)
(=>
(g2 c1o7)
(f c1o13 c1o7)
)
(=>
(g2 c1o8)
(f c1o13 c1o8)
)
(=>
(g2 c1o9)
(f c1o13 c1o9)
)
(=>
(g2 c1o10)
(f c1o13 c1o10)
)
(=>
(g2 c1o11)
(f c1o13 c1o11)
)
(=>
(g2 c1o12)
(f c1o13 c1o12)
)
(=>
(g2 c1o13)
(f c1o13 c1o13)
)
(=>
(g2 c1o14)
(f c1o13 c1o14)
)
(=>
(g2 c1o15)
(f c1o13 c1o15)
)
(=>
(g2 c1o16)
(f c1o13 c1o16)
)
(=>
(g2 c1o17)
(f c1o13 c1o17)
)
(=>
(g2 c1o18)
(f c1o13 c1o18)
)
(=>
(g2 c1o19)
(f c1o13 c1o19)
)
)
)
)
(and
(g1 c1o14)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o14 c1o1)
)
(=>
(g2 c1o2)
(f c1o14 c1o2)
)
(=>
(g2 c1o3)
(f c1o14 c1o3)
)
(=>
(g2 c1o4)
(f c1o14 c1o4)
)
(=>
(g2 c1o5)
(f c1o14 c1o5)
)
(=>
(g2 c1o6)
(f c1o14 c1o6)
)
(=>
(g2 c1o7)
(f c1o14 c1o7)
)
(=>
(g2 c1o8)
(f c1o14 c1o8)
)
(=>
(g2 c1o9)
(f c1o14 c1o9)
)
(=>
(g2 c1o10)
(f c1o14 c1o10)
)
(=>
(g2 c1o11)
(f c1o14 c1o11)
)
(=>
(g2 c1o12)
(f c1o14 c1o12)
)
(=>
(g2 c1o13)
(f c1o14 c1o13)
)
(=>
(g2 c1o14)
(f c1o14 c1o14)
)
(=>
(g2 c1o15)
(f c1o14 c1o15)
)
(=>
(g2 c1o16)
(f c1o14 c1o16)
)
(=>
(g2 c1o17)
(f c1o14 c1o17)
)
(=>
(g2 c1o18)
(f c1o14 c1o18)
)
(=>
(g2 c1o19)
(f c1o14 c1o19)
)
)
)
)
(and
(g1 c1o15)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o15 c1o1)
)
(=>
(g2 c1o2)
(f c1o15 c1o2)
)
(=>
(g2 c1o3)
(f c1o15 c1o3)
)
(=>
(g2 c1o4)
(f c1o15 c1o4)
)
(=>
(g2 c1o5)
(f c1o15 c1o5)
)
(=>
(g2 c1o6)
(f c1o15 c1o6)
)
(=>
(g2 c1o7)
(f c1o15 c1o7)
)
(=>
(g2 c1o8)
(f c1o15 c1o8)
)
(=>
(g2 c1o9)
(f c1o15 c1o9)
)
(=>
(g2 c1o10)
(f c1o15 c1o10)
)
(=>
(g2 c1o11)
(f c1o15 c1o11)
)
(=>
(g2 c1o12)
(f c1o15 c1o12)
)
(=>
(g2 c1o13)
(f c1o15 c1o13)
)
(=>
(g2 c1o14)
(f c1o15 c1o14)
)
(=>
(g2 c1o15)
(f c1o15 c1o15)
)
(=>
(g2 c1o16)
(f c1o15 c1o16)
)
(=>
(g2 c1o17)
(f c1o15 c1o17)
)
(=>
(g2 c1o18)
(f c1o15 c1o18)
)
(=>
(g2 c1o19)
(f c1o15 c1o19)
)
)
)
)
(and
(g1 c1o16)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o16 c1o1)
)
(=>
(g2 c1o2)
(f c1o16 c1o2)
)
(=>
(g2 c1o3)
(f c1o16 c1o3)
)
(=>
(g2 c1o4)
(f c1o16 c1o4)
)
(=>
(g2 c1o5)
(f c1o16 c1o5)
)
(=>
(g2 c1o6)
(f c1o16 c1o6)
)
(=>
(g2 c1o7)
(f c1o16 c1o7)
)
(=>
(g2 c1o8)
(f c1o16 c1o8)
)
(=>
(g2 c1o9)
(f c1o16 c1o9)
)
(=>
(g2 c1o10)
(f c1o16 c1o10)
)
(=>
(g2 c1o11)
(f c1o16 c1o11)
)
(=>
(g2 c1o12)
(f c1o16 c1o12)
)
(=>
(g2 c1o13)
(f c1o16 c1o13)
)
(=>
(g2 c1o14)
(f c1o16 c1o14)
)
(=>
(g2 c1o15)
(f c1o16 c1o15)
)
(=>
(g2 c1o16)
(f c1o16 c1o16)
)
(=>
(g2 c1o17)
(f c1o16 c1o17)
)
(=>
(g2 c1o18)
(f c1o16 c1o18)
)
(=>
(g2 c1o19)
(f c1o16 c1o19)
)
)
)
)
(and
(g1 c1o17)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o17 c1o1)
)
(=>
(g2 c1o2)
(f c1o17 c1o2)
)
(=>
(g2 c1o3)
(f c1o17 c1o3)
)
(=>
(g2 c1o4)
(f c1o17 c1o4)
)
(=>
(g2 c1o5)
(f c1o17 c1o5)
)
(=>
(g2 c1o6)
(f c1o17 c1o6)
)
(=>
(g2 c1o7)
(f c1o17 c1o7)
)
(=>
(g2 c1o8)
(f c1o17 c1o8)
)
(=>
(g2 c1o9)
(f c1o17 c1o9)
)
(=>
(g2 c1o10)
(f c1o17 c1o10)
)
(=>
(g2 c1o11)
(f c1o17 c1o11)
)
(=>
(g2 c1o12)
(f c1o17 c1o12)
)
(=>
(g2 c1o13)
(f c1o17 c1o13)
)
(=>
(g2 c1o14)
(f c1o17 c1o14)
)
(=>
(g2 c1o15)
(f c1o17 c1o15)
)
(=>
(g2 c1o16)
(f c1o17 c1o16)
)
(=>
(g2 c1o17)
(f c1o17 c1o17)
)
(=>
(g2 c1o18)
(f c1o17 c1o18)
)
(=>
(g2 c1o19)
(f c1o17 c1o19)
)
)
)
)
(and
(g1 c1o18)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o18 c1o1)
)
(=>
(g2 c1o2)
(f c1o18 c1o2)
)
(=>
(g2 c1o3)
(f c1o18 c1o3)
)
(=>
(g2 c1o4)
(f c1o18 c1o4)
)
(=>
(g2 c1o5)
(f c1o18 c1o5)
)
(=>
(g2 c1o6)
(f c1o18 c1o6)
)
(=>
(g2 c1o7)
(f c1o18 c1o7)
)
(=>
(g2 c1o8)
(f c1o18 c1o8)
)
(=>
(g2 c1o9)
(f c1o18 c1o9)
)
(=>
(g2 c1o10)
(f c1o18 c1o10)
)
(=>
(g2 c1o11)
(f c1o18 c1o11)
)
(=>
(g2 c1o12)
(f c1o18 c1o12)
)
(=>
(g2 c1o13)
(f c1o18 c1o13)
)
(=>
(g2 c1o14)
(f c1o18 c1o14)
)
(=>
(g2 c1o15)
(f c1o18 c1o15)
)
(=>
(g2 c1o16)
(f c1o18 c1o16)
)
(=>
(g2 c1o17)
(f c1o18 c1o17)
)
(=>
(g2 c1o18)
(f c1o18 c1o18)
)
(=>
(g2 c1o19)
(f c1o18 c1o19)
)
)
)
)
(and
(g1 c1o19)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
(g2 c1o4)
(g2 c1o5)
(g2 c1o6)
(g2 c1o7)
(g2 c1o8)
(g2 c1o9)
(g2 c1o10)
(g2 c1o11)
(g2 c1o12)
(g2 c1o13)
(g2 c1o14)
(g2 c1o15)
(g2 c1o16)
(g2 c1o17)
(g2 c1o18)
(g2 c1o19)
)
(and
(=>
(g2 c1o1)
(f c1o19 c1o1)
)
(=>
(g2 c1o2)
(f c1o19 c1o2)
)
(=>
(g2 c1o3)
(f c1o19 c1o3)
)
(=>
(g2 c1o4)
(f c1o19 c1o4)
)
(=>
(g2 c1o5)
(f c1o19 c1o5)
)
(=>
(g2 c1o6)
(f c1o19 c1o6)
)
(=>
(g2 c1o7)
(f c1o19 c1o7)
)
(=>
(g2 c1o8)
(f c1o19 c1o8)
)
(=>
(g2 c1o9)
(f c1o19 c1o9)
)
(=>
(g2 c1o10)
(f c1o19 c1o10)
)
(=>
(g2 c1o11)
(f c1o19 c1o11)
)
(=>
(g2 c1o12)
(f c1o19 c1o12)
)
(=>
(g2 c1o13)
(f c1o19 c1o13)
)
(=>
(g2 c1o14)
(f c1o19 c1o14)
)
(=>
(g2 c1o15)
(f c1o19 c1o15)
)
(=>
(g2 c1o16)
(f c1o19 c1o16)
)
(=>
(g2 c1o17)
(f c1o19 c1o17)
)
(=>
(g2 c1o18)
(f c1o19 c1o18)
)
(=>
(g2 c1o19)
(f c1o19 c1o19)
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
(and
(g2 c1o6)
(f c1o1 c1o6)
)
(and
(g2 c1o7)
(f c1o1 c1o7)
)
(and
(g2 c1o8)
(f c1o1 c1o8)
)
(and
(g2 c1o9)
(f c1o1 c1o9)
)
(and
(g2 c1o10)
(f c1o1 c1o10)
)
(and
(g2 c1o11)
(f c1o1 c1o11)
)
(and
(g2 c1o12)
(f c1o1 c1o12)
)
(and
(g2 c1o13)
(f c1o1 c1o13)
)
(and
(g2 c1o14)
(f c1o1 c1o14)
)
(and
(g2 c1o15)
(f c1o1 c1o15)
)
(and
(g2 c1o16)
(f c1o1 c1o16)
)
(and
(g2 c1o17)
(f c1o1 c1o17)
)
(and
(g2 c1o18)
(f c1o1 c1o18)
)
(and
(g2 c1o19)
(f c1o1 c1o19)
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
(and
(g2 c1o6)
(f c1o2 c1o6)
)
(and
(g2 c1o7)
(f c1o2 c1o7)
)
(and
(g2 c1o8)
(f c1o2 c1o8)
)
(and
(g2 c1o9)
(f c1o2 c1o9)
)
(and
(g2 c1o10)
(f c1o2 c1o10)
)
(and
(g2 c1o11)
(f c1o2 c1o11)
)
(and
(g2 c1o12)
(f c1o2 c1o12)
)
(and
(g2 c1o13)
(f c1o2 c1o13)
)
(and
(g2 c1o14)
(f c1o2 c1o14)
)
(and
(g2 c1o15)
(f c1o2 c1o15)
)
(and
(g2 c1o16)
(f c1o2 c1o16)
)
(and
(g2 c1o17)
(f c1o2 c1o17)
)
(and
(g2 c1o18)
(f c1o2 c1o18)
)
(and
(g2 c1o19)
(f c1o2 c1o19)
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
(and
(g2 c1o6)
(f c1o3 c1o6)
)
(and
(g2 c1o7)
(f c1o3 c1o7)
)
(and
(g2 c1o8)
(f c1o3 c1o8)
)
(and
(g2 c1o9)
(f c1o3 c1o9)
)
(and
(g2 c1o10)
(f c1o3 c1o10)
)
(and
(g2 c1o11)
(f c1o3 c1o11)
)
(and
(g2 c1o12)
(f c1o3 c1o12)
)
(and
(g2 c1o13)
(f c1o3 c1o13)
)
(and
(g2 c1o14)
(f c1o3 c1o14)
)
(and
(g2 c1o15)
(f c1o3 c1o15)
)
(and
(g2 c1o16)
(f c1o3 c1o16)
)
(and
(g2 c1o17)
(f c1o3 c1o17)
)
(and
(g2 c1o18)
(f c1o3 c1o18)
)
(and
(g2 c1o19)
(f c1o3 c1o19)
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
(and
(g2 c1o6)
(f c1o4 c1o6)
)
(and
(g2 c1o7)
(f c1o4 c1o7)
)
(and
(g2 c1o8)
(f c1o4 c1o8)
)
(and
(g2 c1o9)
(f c1o4 c1o9)
)
(and
(g2 c1o10)
(f c1o4 c1o10)
)
(and
(g2 c1o11)
(f c1o4 c1o11)
)
(and
(g2 c1o12)
(f c1o4 c1o12)
)
(and
(g2 c1o13)
(f c1o4 c1o13)
)
(and
(g2 c1o14)
(f c1o4 c1o14)
)
(and
(g2 c1o15)
(f c1o4 c1o15)
)
(and
(g2 c1o16)
(f c1o4 c1o16)
)
(and
(g2 c1o17)
(f c1o4 c1o17)
)
(and
(g2 c1o18)
(f c1o4 c1o18)
)
(and
(g2 c1o19)
(f c1o4 c1o19)
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
(and
(g2 c1o6)
(f c1o5 c1o6)
)
(and
(g2 c1o7)
(f c1o5 c1o7)
)
(and
(g2 c1o8)
(f c1o5 c1o8)
)
(and
(g2 c1o9)
(f c1o5 c1o9)
)
(and
(g2 c1o10)
(f c1o5 c1o10)
)
(and
(g2 c1o11)
(f c1o5 c1o11)
)
(and
(g2 c1o12)
(f c1o5 c1o12)
)
(and
(g2 c1o13)
(f c1o5 c1o13)
)
(and
(g2 c1o14)
(f c1o5 c1o14)
)
(and
(g2 c1o15)
(f c1o5 c1o15)
)
(and
(g2 c1o16)
(f c1o5 c1o16)
)
(and
(g2 c1o17)
(f c1o5 c1o17)
)
(and
(g2 c1o18)
(f c1o5 c1o18)
)
(and
(g2 c1o19)
(f c1o5 c1o19)
)
)
)
(and
(g1 c1o6)
(or
(and
(g2 c1o1)
(f c1o6 c1o1)
)
(and
(g2 c1o2)
(f c1o6 c1o2)
)
(and
(g2 c1o3)
(f c1o6 c1o3)
)
(and
(g2 c1o4)
(f c1o6 c1o4)
)
(and
(g2 c1o5)
(f c1o6 c1o5)
)
(and
(g2 c1o6)
(f c1o6 c1o6)
)
(and
(g2 c1o7)
(f c1o6 c1o7)
)
(and
(g2 c1o8)
(f c1o6 c1o8)
)
(and
(g2 c1o9)
(f c1o6 c1o9)
)
(and
(g2 c1o10)
(f c1o6 c1o10)
)
(and
(g2 c1o11)
(f c1o6 c1o11)
)
(and
(g2 c1o12)
(f c1o6 c1o12)
)
(and
(g2 c1o13)
(f c1o6 c1o13)
)
(and
(g2 c1o14)
(f c1o6 c1o14)
)
(and
(g2 c1o15)
(f c1o6 c1o15)
)
(and
(g2 c1o16)
(f c1o6 c1o16)
)
(and
(g2 c1o17)
(f c1o6 c1o17)
)
(and
(g2 c1o18)
(f c1o6 c1o18)
)
(and
(g2 c1o19)
(f c1o6 c1o19)
)
)
)
(and
(g1 c1o7)
(or
(and
(g2 c1o1)
(f c1o7 c1o1)
)
(and
(g2 c1o2)
(f c1o7 c1o2)
)
(and
(g2 c1o3)
(f c1o7 c1o3)
)
(and
(g2 c1o4)
(f c1o7 c1o4)
)
(and
(g2 c1o5)
(f c1o7 c1o5)
)
(and
(g2 c1o6)
(f c1o7 c1o6)
)
(and
(g2 c1o7)
(f c1o7 c1o7)
)
(and
(g2 c1o8)
(f c1o7 c1o8)
)
(and
(g2 c1o9)
(f c1o7 c1o9)
)
(and
(g2 c1o10)
(f c1o7 c1o10)
)
(and
(g2 c1o11)
(f c1o7 c1o11)
)
(and
(g2 c1o12)
(f c1o7 c1o12)
)
(and
(g2 c1o13)
(f c1o7 c1o13)
)
(and
(g2 c1o14)
(f c1o7 c1o14)
)
(and
(g2 c1o15)
(f c1o7 c1o15)
)
(and
(g2 c1o16)
(f c1o7 c1o16)
)
(and
(g2 c1o17)
(f c1o7 c1o17)
)
(and
(g2 c1o18)
(f c1o7 c1o18)
)
(and
(g2 c1o19)
(f c1o7 c1o19)
)
)
)
(and
(g1 c1o8)
(or
(and
(g2 c1o1)
(f c1o8 c1o1)
)
(and
(g2 c1o2)
(f c1o8 c1o2)
)
(and
(g2 c1o3)
(f c1o8 c1o3)
)
(and
(g2 c1o4)
(f c1o8 c1o4)
)
(and
(g2 c1o5)
(f c1o8 c1o5)
)
(and
(g2 c1o6)
(f c1o8 c1o6)
)
(and
(g2 c1o7)
(f c1o8 c1o7)
)
(and
(g2 c1o8)
(f c1o8 c1o8)
)
(and
(g2 c1o9)
(f c1o8 c1o9)
)
(and
(g2 c1o10)
(f c1o8 c1o10)
)
(and
(g2 c1o11)
(f c1o8 c1o11)
)
(and
(g2 c1o12)
(f c1o8 c1o12)
)
(and
(g2 c1o13)
(f c1o8 c1o13)
)
(and
(g2 c1o14)
(f c1o8 c1o14)
)
(and
(g2 c1o15)
(f c1o8 c1o15)
)
(and
(g2 c1o16)
(f c1o8 c1o16)
)
(and
(g2 c1o17)
(f c1o8 c1o17)
)
(and
(g2 c1o18)
(f c1o8 c1o18)
)
(and
(g2 c1o19)
(f c1o8 c1o19)
)
)
)
(and
(g1 c1o9)
(or
(and
(g2 c1o1)
(f c1o9 c1o1)
)
(and
(g2 c1o2)
(f c1o9 c1o2)
)
(and
(g2 c1o3)
(f c1o9 c1o3)
)
(and
(g2 c1o4)
(f c1o9 c1o4)
)
(and
(g2 c1o5)
(f c1o9 c1o5)
)
(and
(g2 c1o6)
(f c1o9 c1o6)
)
(and
(g2 c1o7)
(f c1o9 c1o7)
)
(and
(g2 c1o8)
(f c1o9 c1o8)
)
(and
(g2 c1o9)
(f c1o9 c1o9)
)
(and
(g2 c1o10)
(f c1o9 c1o10)
)
(and
(g2 c1o11)
(f c1o9 c1o11)
)
(and
(g2 c1o12)
(f c1o9 c1o12)
)
(and
(g2 c1o13)
(f c1o9 c1o13)
)
(and
(g2 c1o14)
(f c1o9 c1o14)
)
(and
(g2 c1o15)
(f c1o9 c1o15)
)
(and
(g2 c1o16)
(f c1o9 c1o16)
)
(and
(g2 c1o17)
(f c1o9 c1o17)
)
(and
(g2 c1o18)
(f c1o9 c1o18)
)
(and
(g2 c1o19)
(f c1o9 c1o19)
)
)
)
(and
(g1 c1o10)
(or
(and
(g2 c1o1)
(f c1o10 c1o1)
)
(and
(g2 c1o2)
(f c1o10 c1o2)
)
(and
(g2 c1o3)
(f c1o10 c1o3)
)
(and
(g2 c1o4)
(f c1o10 c1o4)
)
(and
(g2 c1o5)
(f c1o10 c1o5)
)
(and
(g2 c1o6)
(f c1o10 c1o6)
)
(and
(g2 c1o7)
(f c1o10 c1o7)
)
(and
(g2 c1o8)
(f c1o10 c1o8)
)
(and
(g2 c1o9)
(f c1o10 c1o9)
)
(and
(g2 c1o10)
(f c1o10 c1o10)
)
(and
(g2 c1o11)
(f c1o10 c1o11)
)
(and
(g2 c1o12)
(f c1o10 c1o12)
)
(and
(g2 c1o13)
(f c1o10 c1o13)
)
(and
(g2 c1o14)
(f c1o10 c1o14)
)
(and
(g2 c1o15)
(f c1o10 c1o15)
)
(and
(g2 c1o16)
(f c1o10 c1o16)
)
(and
(g2 c1o17)
(f c1o10 c1o17)
)
(and
(g2 c1o18)
(f c1o10 c1o18)
)
(and
(g2 c1o19)
(f c1o10 c1o19)
)
)
)
(and
(g1 c1o11)
(or
(and
(g2 c1o1)
(f c1o11 c1o1)
)
(and
(g2 c1o2)
(f c1o11 c1o2)
)
(and
(g2 c1o3)
(f c1o11 c1o3)
)
(and
(g2 c1o4)
(f c1o11 c1o4)
)
(and
(g2 c1o5)
(f c1o11 c1o5)
)
(and
(g2 c1o6)
(f c1o11 c1o6)
)
(and
(g2 c1o7)
(f c1o11 c1o7)
)
(and
(g2 c1o8)
(f c1o11 c1o8)
)
(and
(g2 c1o9)
(f c1o11 c1o9)
)
(and
(g2 c1o10)
(f c1o11 c1o10)
)
(and
(g2 c1o11)
(f c1o11 c1o11)
)
(and
(g2 c1o12)
(f c1o11 c1o12)
)
(and
(g2 c1o13)
(f c1o11 c1o13)
)
(and
(g2 c1o14)
(f c1o11 c1o14)
)
(and
(g2 c1o15)
(f c1o11 c1o15)
)
(and
(g2 c1o16)
(f c1o11 c1o16)
)
(and
(g2 c1o17)
(f c1o11 c1o17)
)
(and
(g2 c1o18)
(f c1o11 c1o18)
)
(and
(g2 c1o19)
(f c1o11 c1o19)
)
)
)
(and
(g1 c1o12)
(or
(and
(g2 c1o1)
(f c1o12 c1o1)
)
(and
(g2 c1o2)
(f c1o12 c1o2)
)
(and
(g2 c1o3)
(f c1o12 c1o3)
)
(and
(g2 c1o4)
(f c1o12 c1o4)
)
(and
(g2 c1o5)
(f c1o12 c1o5)
)
(and
(g2 c1o6)
(f c1o12 c1o6)
)
(and
(g2 c1o7)
(f c1o12 c1o7)
)
(and
(g2 c1o8)
(f c1o12 c1o8)
)
(and
(g2 c1o9)
(f c1o12 c1o9)
)
(and
(g2 c1o10)
(f c1o12 c1o10)
)
(and
(g2 c1o11)
(f c1o12 c1o11)
)
(and
(g2 c1o12)
(f c1o12 c1o12)
)
(and
(g2 c1o13)
(f c1o12 c1o13)
)
(and
(g2 c1o14)
(f c1o12 c1o14)
)
(and
(g2 c1o15)
(f c1o12 c1o15)
)
(and
(g2 c1o16)
(f c1o12 c1o16)
)
(and
(g2 c1o17)
(f c1o12 c1o17)
)
(and
(g2 c1o18)
(f c1o12 c1o18)
)
(and
(g2 c1o19)
(f c1o12 c1o19)
)
)
)
(and
(g1 c1o13)
(or
(and
(g2 c1o1)
(f c1o13 c1o1)
)
(and
(g2 c1o2)
(f c1o13 c1o2)
)
(and
(g2 c1o3)
(f c1o13 c1o3)
)
(and
(g2 c1o4)
(f c1o13 c1o4)
)
(and
(g2 c1o5)
(f c1o13 c1o5)
)
(and
(g2 c1o6)
(f c1o13 c1o6)
)
(and
(g2 c1o7)
(f c1o13 c1o7)
)
(and
(g2 c1o8)
(f c1o13 c1o8)
)
(and
(g2 c1o9)
(f c1o13 c1o9)
)
(and
(g2 c1o10)
(f c1o13 c1o10)
)
(and
(g2 c1o11)
(f c1o13 c1o11)
)
(and
(g2 c1o12)
(f c1o13 c1o12)
)
(and
(g2 c1o13)
(f c1o13 c1o13)
)
(and
(g2 c1o14)
(f c1o13 c1o14)
)
(and
(g2 c1o15)
(f c1o13 c1o15)
)
(and
(g2 c1o16)
(f c1o13 c1o16)
)
(and
(g2 c1o17)
(f c1o13 c1o17)
)
(and
(g2 c1o18)
(f c1o13 c1o18)
)
(and
(g2 c1o19)
(f c1o13 c1o19)
)
)
)
(and
(g1 c1o14)
(or
(and
(g2 c1o1)
(f c1o14 c1o1)
)
(and
(g2 c1o2)
(f c1o14 c1o2)
)
(and
(g2 c1o3)
(f c1o14 c1o3)
)
(and
(g2 c1o4)
(f c1o14 c1o4)
)
(and
(g2 c1o5)
(f c1o14 c1o5)
)
(and
(g2 c1o6)
(f c1o14 c1o6)
)
(and
(g2 c1o7)
(f c1o14 c1o7)
)
(and
(g2 c1o8)
(f c1o14 c1o8)
)
(and
(g2 c1o9)
(f c1o14 c1o9)
)
(and
(g2 c1o10)
(f c1o14 c1o10)
)
(and
(g2 c1o11)
(f c1o14 c1o11)
)
(and
(g2 c1o12)
(f c1o14 c1o12)
)
(and
(g2 c1o13)
(f c1o14 c1o13)
)
(and
(g2 c1o14)
(f c1o14 c1o14)
)
(and
(g2 c1o15)
(f c1o14 c1o15)
)
(and
(g2 c1o16)
(f c1o14 c1o16)
)
(and
(g2 c1o17)
(f c1o14 c1o17)
)
(and
(g2 c1o18)
(f c1o14 c1o18)
)
(and
(g2 c1o19)
(f c1o14 c1o19)
)
)
)
(and
(g1 c1o15)
(or
(and
(g2 c1o1)
(f c1o15 c1o1)
)
(and
(g2 c1o2)
(f c1o15 c1o2)
)
(and
(g2 c1o3)
(f c1o15 c1o3)
)
(and
(g2 c1o4)
(f c1o15 c1o4)
)
(and
(g2 c1o5)
(f c1o15 c1o5)
)
(and
(g2 c1o6)
(f c1o15 c1o6)
)
(and
(g2 c1o7)
(f c1o15 c1o7)
)
(and
(g2 c1o8)
(f c1o15 c1o8)
)
(and
(g2 c1o9)
(f c1o15 c1o9)
)
(and
(g2 c1o10)
(f c1o15 c1o10)
)
(and
(g2 c1o11)
(f c1o15 c1o11)
)
(and
(g2 c1o12)
(f c1o15 c1o12)
)
(and
(g2 c1o13)
(f c1o15 c1o13)
)
(and
(g2 c1o14)
(f c1o15 c1o14)
)
(and
(g2 c1o15)
(f c1o15 c1o15)
)
(and
(g2 c1o16)
(f c1o15 c1o16)
)
(and
(g2 c1o17)
(f c1o15 c1o17)
)
(and
(g2 c1o18)
(f c1o15 c1o18)
)
(and
(g2 c1o19)
(f c1o15 c1o19)
)
)
)
(and
(g1 c1o16)
(or
(and
(g2 c1o1)
(f c1o16 c1o1)
)
(and
(g2 c1o2)
(f c1o16 c1o2)
)
(and
(g2 c1o3)
(f c1o16 c1o3)
)
(and
(g2 c1o4)
(f c1o16 c1o4)
)
(and
(g2 c1o5)
(f c1o16 c1o5)
)
(and
(g2 c1o6)
(f c1o16 c1o6)
)
(and
(g2 c1o7)
(f c1o16 c1o7)
)
(and
(g2 c1o8)
(f c1o16 c1o8)
)
(and
(g2 c1o9)
(f c1o16 c1o9)
)
(and
(g2 c1o10)
(f c1o16 c1o10)
)
(and
(g2 c1o11)
(f c1o16 c1o11)
)
(and
(g2 c1o12)
(f c1o16 c1o12)
)
(and
(g2 c1o13)
(f c1o16 c1o13)
)
(and
(g2 c1o14)
(f c1o16 c1o14)
)
(and
(g2 c1o15)
(f c1o16 c1o15)
)
(and
(g2 c1o16)
(f c1o16 c1o16)
)
(and
(g2 c1o17)
(f c1o16 c1o17)
)
(and
(g2 c1o18)
(f c1o16 c1o18)
)
(and
(g2 c1o19)
(f c1o16 c1o19)
)
)
)
(and
(g1 c1o17)
(or
(and
(g2 c1o1)
(f c1o17 c1o1)
)
(and
(g2 c1o2)
(f c1o17 c1o2)
)
(and
(g2 c1o3)
(f c1o17 c1o3)
)
(and
(g2 c1o4)
(f c1o17 c1o4)
)
(and
(g2 c1o5)
(f c1o17 c1o5)
)
(and
(g2 c1o6)
(f c1o17 c1o6)
)
(and
(g2 c1o7)
(f c1o17 c1o7)
)
(and
(g2 c1o8)
(f c1o17 c1o8)
)
(and
(g2 c1o9)
(f c1o17 c1o9)
)
(and
(g2 c1o10)
(f c1o17 c1o10)
)
(and
(g2 c1o11)
(f c1o17 c1o11)
)
(and
(g2 c1o12)
(f c1o17 c1o12)
)
(and
(g2 c1o13)
(f c1o17 c1o13)
)
(and
(g2 c1o14)
(f c1o17 c1o14)
)
(and
(g2 c1o15)
(f c1o17 c1o15)
)
(and
(g2 c1o16)
(f c1o17 c1o16)
)
(and
(g2 c1o17)
(f c1o17 c1o17)
)
(and
(g2 c1o18)
(f c1o17 c1o18)
)
(and
(g2 c1o19)
(f c1o17 c1o19)
)
)
)
(and
(g1 c1o18)
(or
(and
(g2 c1o1)
(f c1o18 c1o1)
)
(and
(g2 c1o2)
(f c1o18 c1o2)
)
(and
(g2 c1o3)
(f c1o18 c1o3)
)
(and
(g2 c1o4)
(f c1o18 c1o4)
)
(and
(g2 c1o5)
(f c1o18 c1o5)
)
(and
(g2 c1o6)
(f c1o18 c1o6)
)
(and
(g2 c1o7)
(f c1o18 c1o7)
)
(and
(g2 c1o8)
(f c1o18 c1o8)
)
(and
(g2 c1o9)
(f c1o18 c1o9)
)
(and
(g2 c1o10)
(f c1o18 c1o10)
)
(and
(g2 c1o11)
(f c1o18 c1o11)
)
(and
(g2 c1o12)
(f c1o18 c1o12)
)
(and
(g2 c1o13)
(f c1o18 c1o13)
)
(and
(g2 c1o14)
(f c1o18 c1o14)
)
(and
(g2 c1o15)
(f c1o18 c1o15)
)
(and
(g2 c1o16)
(f c1o18 c1o16)
)
(and
(g2 c1o17)
(f c1o18 c1o17)
)
(and
(g2 c1o18)
(f c1o18 c1o18)
)
(and
(g2 c1o19)
(f c1o18 c1o19)
)
)
)
(and
(g1 c1o19)
(or
(and
(g2 c1o1)
(f c1o19 c1o1)
)
(and
(g2 c1o2)
(f c1o19 c1o2)
)
(and
(g2 c1o3)
(f c1o19 c1o3)
)
(and
(g2 c1o4)
(f c1o19 c1o4)
)
(and
(g2 c1o5)
(f c1o19 c1o5)
)
(and
(g2 c1o6)
(f c1o19 c1o6)
)
(and
(g2 c1o7)
(f c1o19 c1o7)
)
(and
(g2 c1o8)
(f c1o19 c1o8)
)
(and
(g2 c1o9)
(f c1o19 c1o9)
)
(and
(g2 c1o10)
(f c1o19 c1o10)
)
(and
(g2 c1o11)
(f c1o19 c1o11)
)
(and
(g2 c1o12)
(f c1o19 c1o12)
)
(and
(g2 c1o13)
(f c1o19 c1o13)
)
(and
(g2 c1o14)
(f c1o19 c1o14)
)
(and
(g2 c1o15)
(f c1o19 c1o15)
)
(and
(g2 c1o16)
(f c1o19 c1o16)
)
(and
(g2 c1o17)
(f c1o19 c1o17)
)
(and
(g2 c1o18)
(f c1o19 c1o18)
)
(and
(g2 c1o19)
(f c1o19 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o1 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o1 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o1 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o1 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o1 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o1 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o1 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o1 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o1 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o1 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o1 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o1 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o1 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o1 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o2 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o2 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o2 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o2 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o2 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o2 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o2 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o2 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o2 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o2 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o2 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o2 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o2 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o2 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o3 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o3 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o3 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o3 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o3 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o3 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o3 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o3 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o3 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o3 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o3 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o3 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o3 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o3 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o4 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o4 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o4 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o4 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o4 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o4 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o4 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o4 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o4 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o4 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o4 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o4 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o4 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o4 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o5 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o5 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o5 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o5 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o5 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o5 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o5 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o5 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o5 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o5 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o5 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o5 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o5 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o5 c1o19)
)
)
)
)
(and
(g1 c1o6)
(or
(and
(g2 c1o1)
(not
(f c1o6 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o6 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o6 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o6 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o6 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o6 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o6 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o6 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o6 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o6 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o6 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o6 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o6 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o6 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o6 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o6 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o6 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o6 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o6 c1o19)
)
)
)
)
(and
(g1 c1o7)
(or
(and
(g2 c1o1)
(not
(f c1o7 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o7 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o7 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o7 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o7 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o7 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o7 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o7 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o7 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o7 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o7 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o7 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o7 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o7 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o7 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o7 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o7 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o7 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o7 c1o19)
)
)
)
)
(and
(g1 c1o8)
(or
(and
(g2 c1o1)
(not
(f c1o8 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o8 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o8 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o8 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o8 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o8 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o8 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o8 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o8 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o8 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o8 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o8 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o8 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o8 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o8 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o8 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o8 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o8 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o8 c1o19)
)
)
)
)
(and
(g1 c1o9)
(or
(and
(g2 c1o1)
(not
(f c1o9 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o9 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o9 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o9 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o9 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o9 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o9 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o9 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o9 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o9 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o9 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o9 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o9 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o9 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o9 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o9 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o9 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o9 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o9 c1o19)
)
)
)
)
(and
(g1 c1o10)
(or
(and
(g2 c1o1)
(not
(f c1o10 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o10 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o10 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o10 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o10 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o10 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o10 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o10 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o10 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o10 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o10 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o10 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o10 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o10 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o10 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o10 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o10 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o10 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o10 c1o19)
)
)
)
)
(and
(g1 c1o11)
(or
(and
(g2 c1o1)
(not
(f c1o11 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o11 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o11 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o11 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o11 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o11 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o11 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o11 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o11 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o11 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o11 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o11 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o11 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o11 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o11 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o11 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o11 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o11 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o11 c1o19)
)
)
)
)
(and
(g1 c1o12)
(or
(and
(g2 c1o1)
(not
(f c1o12 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o12 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o12 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o12 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o12 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o12 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o12 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o12 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o12 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o12 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o12 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o12 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o12 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o12 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o12 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o12 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o12 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o12 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o12 c1o19)
)
)
)
)
(and
(g1 c1o13)
(or
(and
(g2 c1o1)
(not
(f c1o13 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o13 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o13 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o13 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o13 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o13 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o13 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o13 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o13 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o13 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o13 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o13 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o13 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o13 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o13 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o13 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o13 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o13 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o13 c1o19)
)
)
)
)
(and
(g1 c1o14)
(or
(and
(g2 c1o1)
(not
(f c1o14 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o14 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o14 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o14 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o14 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o14 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o14 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o14 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o14 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o14 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o14 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o14 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o14 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o14 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o14 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o14 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o14 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o14 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o14 c1o19)
)
)
)
)
(and
(g1 c1o15)
(or
(and
(g2 c1o1)
(not
(f c1o15 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o15 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o15 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o15 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o15 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o15 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o15 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o15 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o15 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o15 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o15 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o15 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o15 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o15 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o15 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o15 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o15 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o15 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o15 c1o19)
)
)
)
)
(and
(g1 c1o16)
(or
(and
(g2 c1o1)
(not
(f c1o16 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o16 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o16 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o16 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o16 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o16 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o16 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o16 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o16 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o16 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o16 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o16 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o16 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o16 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o16 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o16 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o16 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o16 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o16 c1o19)
)
)
)
)
(and
(g1 c1o17)
(or
(and
(g2 c1o1)
(not
(f c1o17 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o17 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o17 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o17 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o17 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o17 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o17 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o17 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o17 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o17 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o17 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o17 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o17 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o17 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o17 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o17 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o17 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o17 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o17 c1o19)
)
)
)
)
(and
(g1 c1o18)
(or
(and
(g2 c1o1)
(not
(f c1o18 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o18 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o18 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o18 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o18 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o18 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o18 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o18 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o18 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o18 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o18 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o18 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o18 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o18 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o18 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o18 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o18 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o18 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o18 c1o19)
)
)
)
)
(and
(g1 c1o19)
(or
(and
(g2 c1o1)
(not
(f c1o19 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o19 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o19 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o19 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o19 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o19 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o19 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o19 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o19 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o19 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o19 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o19 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o19 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o19 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o19 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o19 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o19 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o19 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o19 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o1 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o1 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o1 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o1 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o1 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o1 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o1 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o1 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o1 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o1 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o1 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o1 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o1 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o1 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o2 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o2 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o2 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o2 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o2 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o2 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o2 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o2 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o2 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o2 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o2 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o2 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o2 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o2 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o3 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o3 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o3 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o3 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o3 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o3 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o3 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o3 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o3 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o3 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o3 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o3 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o3 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o3 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o4 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o4 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o4 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o4 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o4 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o4 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o4 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o4 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o4 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o4 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o4 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o4 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o4 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o4 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o5 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o5 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o5 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o5 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o5 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o5 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o5 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o5 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o5 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o5 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o5 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o5 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o5 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o5 c1o19)
)
)
)
)
(and
(g1 c1o6)
(and
(=>
(g2 c1o1)
(not
(f c1o6 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o6 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o6 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o6 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o6 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o6 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o6 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o6 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o6 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o6 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o6 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o6 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o6 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o6 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o6 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o6 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o6 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o6 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o6 c1o19)
)
)
)
)
(and
(g1 c1o7)
(and
(=>
(g2 c1o1)
(not
(f c1o7 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o7 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o7 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o7 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o7 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o7 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o7 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o7 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o7 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o7 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o7 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o7 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o7 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o7 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o7 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o7 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o7 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o7 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o7 c1o19)
)
)
)
)
(and
(g1 c1o8)
(and
(=>
(g2 c1o1)
(not
(f c1o8 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o8 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o8 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o8 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o8 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o8 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o8 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o8 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o8 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o8 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o8 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o8 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o8 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o8 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o8 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o8 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o8 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o8 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o8 c1o19)
)
)
)
)
(and
(g1 c1o9)
(and
(=>
(g2 c1o1)
(not
(f c1o9 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o9 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o9 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o9 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o9 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o9 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o9 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o9 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o9 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o9 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o9 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o9 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o9 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o9 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o9 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o9 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o9 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o9 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o9 c1o19)
)
)
)
)
(and
(g1 c1o10)
(and
(=>
(g2 c1o1)
(not
(f c1o10 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o10 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o10 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o10 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o10 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o10 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o10 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o10 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o10 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o10 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o10 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o10 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o10 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o10 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o10 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o10 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o10 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o10 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o10 c1o19)
)
)
)
)
(and
(g1 c1o11)
(and
(=>
(g2 c1o1)
(not
(f c1o11 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o11 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o11 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o11 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o11 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o11 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o11 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o11 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o11 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o11 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o11 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o11 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o11 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o11 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o11 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o11 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o11 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o11 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o11 c1o19)
)
)
)
)
(and
(g1 c1o12)
(and
(=>
(g2 c1o1)
(not
(f c1o12 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o12 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o12 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o12 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o12 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o12 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o12 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o12 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o12 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o12 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o12 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o12 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o12 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o12 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o12 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o12 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o12 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o12 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o12 c1o19)
)
)
)
)
(and
(g1 c1o13)
(and
(=>
(g2 c1o1)
(not
(f c1o13 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o13 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o13 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o13 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o13 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o13 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o13 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o13 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o13 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o13 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o13 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o13 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o13 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o13 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o13 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o13 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o13 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o13 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o13 c1o19)
)
)
)
)
(and
(g1 c1o14)
(and
(=>
(g2 c1o1)
(not
(f c1o14 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o14 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o14 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o14 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o14 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o14 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o14 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o14 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o14 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o14 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o14 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o14 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o14 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o14 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o14 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o14 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o14 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o14 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o14 c1o19)
)
)
)
)
(and
(g1 c1o15)
(and
(=>
(g2 c1o1)
(not
(f c1o15 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o15 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o15 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o15 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o15 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o15 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o15 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o15 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o15 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o15 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o15 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o15 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o15 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o15 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o15 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o15 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o15 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o15 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o15 c1o19)
)
)
)
)
(and
(g1 c1o16)
(and
(=>
(g2 c1o1)
(not
(f c1o16 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o16 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o16 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o16 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o16 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o16 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o16 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o16 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o16 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o16 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o16 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o16 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o16 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o16 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o16 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o16 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o16 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o16 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o16 c1o19)
)
)
)
)
(and
(g1 c1o17)
(and
(=>
(g2 c1o1)
(not
(f c1o17 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o17 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o17 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o17 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o17 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o17 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o17 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o17 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o17 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o17 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o17 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o17 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o17 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o17 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o17 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o17 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o17 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o17 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o17 c1o19)
)
)
)
)
(and
(g1 c1o18)
(and
(=>
(g2 c1o1)
(not
(f c1o18 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o18 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o18 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o18 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o18 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o18 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o18 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o18 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o18 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o18 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o18 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o18 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o18 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o18 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o18 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o18 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o18 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o18 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o18 c1o19)
)
)
)
)
(and
(g1 c1o19)
(and
(=>
(g2 c1o1)
(not
(f c1o19 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o19 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o19 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o19 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o19 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o19 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o19 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o19 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o19 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o19 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o19 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o19 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o19 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o19 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o19 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o19 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o19 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o19 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o19 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o1 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o1 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o1 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o1 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o1 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o1 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o1 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o1 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o1 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o1 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o1 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o1 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o1 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o1 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o2 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o2 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o2 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o2 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o2 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o2 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o2 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o2 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o2 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o2 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o2 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o2 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o2 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o2 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o3 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o3 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o3 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o3 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o3 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o3 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o3 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o3 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o3 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o3 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o3 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o3 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o3 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o3 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o4 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o4 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o4 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o4 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o4 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o4 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o4 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o4 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o4 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o4 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o4 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o4 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o4 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o4 c1o19)
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
(and
(g2 c1o6)
(not
(f c1o5 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o5 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o5 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o5 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o5 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o5 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o5 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o5 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o5 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o5 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o5 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o5 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o5 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o5 c1o19)
)
)
)
)
(=>
(g1 c1o6)
(or
(and
(g2 c1o1)
(not
(f c1o6 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o6 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o6 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o6 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o6 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o6 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o6 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o6 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o6 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o6 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o6 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o6 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o6 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o6 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o6 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o6 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o6 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o6 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o6 c1o19)
)
)
)
)
(=>
(g1 c1o7)
(or
(and
(g2 c1o1)
(not
(f c1o7 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o7 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o7 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o7 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o7 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o7 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o7 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o7 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o7 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o7 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o7 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o7 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o7 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o7 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o7 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o7 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o7 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o7 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o7 c1o19)
)
)
)
)
(=>
(g1 c1o8)
(or
(and
(g2 c1o1)
(not
(f c1o8 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o8 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o8 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o8 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o8 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o8 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o8 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o8 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o8 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o8 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o8 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o8 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o8 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o8 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o8 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o8 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o8 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o8 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o8 c1o19)
)
)
)
)
(=>
(g1 c1o9)
(or
(and
(g2 c1o1)
(not
(f c1o9 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o9 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o9 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o9 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o9 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o9 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o9 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o9 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o9 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o9 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o9 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o9 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o9 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o9 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o9 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o9 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o9 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o9 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o9 c1o19)
)
)
)
)
(=>
(g1 c1o10)
(or
(and
(g2 c1o1)
(not
(f c1o10 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o10 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o10 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o10 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o10 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o10 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o10 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o10 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o10 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o10 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o10 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o10 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o10 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o10 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o10 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o10 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o10 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o10 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o10 c1o19)
)
)
)
)
(=>
(g1 c1o11)
(or
(and
(g2 c1o1)
(not
(f c1o11 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o11 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o11 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o11 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o11 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o11 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o11 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o11 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o11 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o11 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o11 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o11 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o11 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o11 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o11 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o11 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o11 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o11 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o11 c1o19)
)
)
)
)
(=>
(g1 c1o12)
(or
(and
(g2 c1o1)
(not
(f c1o12 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o12 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o12 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o12 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o12 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o12 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o12 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o12 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o12 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o12 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o12 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o12 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o12 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o12 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o12 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o12 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o12 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o12 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o12 c1o19)
)
)
)
)
(=>
(g1 c1o13)
(or
(and
(g2 c1o1)
(not
(f c1o13 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o13 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o13 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o13 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o13 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o13 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o13 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o13 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o13 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o13 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o13 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o13 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o13 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o13 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o13 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o13 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o13 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o13 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o13 c1o19)
)
)
)
)
(=>
(g1 c1o14)
(or
(and
(g2 c1o1)
(not
(f c1o14 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o14 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o14 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o14 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o14 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o14 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o14 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o14 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o14 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o14 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o14 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o14 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o14 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o14 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o14 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o14 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o14 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o14 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o14 c1o19)
)
)
)
)
(=>
(g1 c1o15)
(or
(and
(g2 c1o1)
(not
(f c1o15 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o15 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o15 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o15 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o15 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o15 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o15 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o15 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o15 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o15 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o15 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o15 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o15 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o15 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o15 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o15 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o15 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o15 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o15 c1o19)
)
)
)
)
(=>
(g1 c1o16)
(or
(and
(g2 c1o1)
(not
(f c1o16 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o16 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o16 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o16 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o16 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o16 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o16 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o16 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o16 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o16 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o16 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o16 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o16 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o16 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o16 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o16 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o16 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o16 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o16 c1o19)
)
)
)
)
(=>
(g1 c1o17)
(or
(and
(g2 c1o1)
(not
(f c1o17 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o17 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o17 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o17 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o17 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o17 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o17 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o17 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o17 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o17 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o17 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o17 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o17 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o17 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o17 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o17 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o17 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o17 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o17 c1o19)
)
)
)
)
(=>
(g1 c1o18)
(or
(and
(g2 c1o1)
(not
(f c1o18 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o18 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o18 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o18 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o18 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o18 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o18 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o18 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o18 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o18 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o18 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o18 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o18 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o18 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o18 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o18 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o18 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o18 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o18 c1o19)
)
)
)
)
(=>
(g1 c1o19)
(or
(and
(g2 c1o1)
(not
(f c1o19 c1o1)
)
)
(and
(g2 c1o2)
(not
(f c1o19 c1o2)
)
)
(and
(g2 c1o3)
(not
(f c1o19 c1o3)
)
)
(and
(g2 c1o4)
(not
(f c1o19 c1o4)
)
)
(and
(g2 c1o5)
(not
(f c1o19 c1o5)
)
)
(and
(g2 c1o6)
(not
(f c1o19 c1o6)
)
)
(and
(g2 c1o7)
(not
(f c1o19 c1o7)
)
)
(and
(g2 c1o8)
(not
(f c1o19 c1o8)
)
)
(and
(g2 c1o9)
(not
(f c1o19 c1o9)
)
)
(and
(g2 c1o10)
(not
(f c1o19 c1o10)
)
)
(and
(g2 c1o11)
(not
(f c1o19 c1o11)
)
)
(and
(g2 c1o12)
(not
(f c1o19 c1o12)
)
)
(and
(g2 c1o13)
(not
(f c1o19 c1o13)
)
)
(and
(g2 c1o14)
(not
(f c1o19 c1o14)
)
)
(and
(g2 c1o15)
(not
(f c1o19 c1o15)
)
)
(and
(g2 c1o16)
(not
(f c1o19 c1o16)
)
)
(and
(g2 c1o17)
(not
(f c1o19 c1o17)
)
)
(and
(g2 c1o18)
(not
(f c1o19 c1o18)
)
)
(and
(g2 c1o19)
(not
(f c1o19 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o1 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o1 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o1 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o1 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o1 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o1 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o1 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o1 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o1 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o1 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o1 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o1 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o1 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o1 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o2 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o2 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o2 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o2 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o2 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o2 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o2 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o2 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o2 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o2 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o2 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o2 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o2 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o2 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o3 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o3 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o3 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o3 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o3 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o3 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o3 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o3 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o3 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o3 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o3 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o3 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o3 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o3 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o4 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o4 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o4 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o4 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o4 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o4 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o4 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o4 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o4 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o4 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o4 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o4 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o4 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o4 c1o19)
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
(=>
(g2 c1o6)
(not
(f c1o5 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o5 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o5 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o5 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o5 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o5 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o5 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o5 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o5 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o5 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o5 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o5 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o5 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o5 c1o19)
)
)
)
)
(=>
(g1 c1o6)
(and
(=>
(g2 c1o1)
(not
(f c1o6 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o6 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o6 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o6 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o6 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o6 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o6 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o6 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o6 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o6 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o6 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o6 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o6 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o6 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o6 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o6 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o6 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o6 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o6 c1o19)
)
)
)
)
(=>
(g1 c1o7)
(and
(=>
(g2 c1o1)
(not
(f c1o7 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o7 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o7 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o7 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o7 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o7 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o7 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o7 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o7 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o7 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o7 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o7 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o7 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o7 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o7 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o7 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o7 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o7 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o7 c1o19)
)
)
)
)
(=>
(g1 c1o8)
(and
(=>
(g2 c1o1)
(not
(f c1o8 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o8 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o8 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o8 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o8 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o8 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o8 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o8 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o8 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o8 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o8 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o8 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o8 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o8 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o8 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o8 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o8 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o8 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o8 c1o19)
)
)
)
)
(=>
(g1 c1o9)
(and
(=>
(g2 c1o1)
(not
(f c1o9 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o9 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o9 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o9 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o9 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o9 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o9 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o9 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o9 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o9 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o9 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o9 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o9 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o9 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o9 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o9 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o9 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o9 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o9 c1o19)
)
)
)
)
(=>
(g1 c1o10)
(and
(=>
(g2 c1o1)
(not
(f c1o10 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o10 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o10 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o10 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o10 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o10 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o10 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o10 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o10 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o10 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o10 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o10 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o10 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o10 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o10 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o10 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o10 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o10 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o10 c1o19)
)
)
)
)
(=>
(g1 c1o11)
(and
(=>
(g2 c1o1)
(not
(f c1o11 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o11 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o11 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o11 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o11 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o11 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o11 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o11 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o11 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o11 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o11 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o11 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o11 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o11 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o11 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o11 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o11 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o11 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o11 c1o19)
)
)
)
)
(=>
(g1 c1o12)
(and
(=>
(g2 c1o1)
(not
(f c1o12 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o12 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o12 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o12 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o12 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o12 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o12 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o12 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o12 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o12 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o12 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o12 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o12 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o12 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o12 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o12 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o12 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o12 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o12 c1o19)
)
)
)
)
(=>
(g1 c1o13)
(and
(=>
(g2 c1o1)
(not
(f c1o13 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o13 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o13 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o13 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o13 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o13 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o13 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o13 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o13 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o13 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o13 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o13 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o13 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o13 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o13 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o13 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o13 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o13 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o13 c1o19)
)
)
)
)
(=>
(g1 c1o14)
(and
(=>
(g2 c1o1)
(not
(f c1o14 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o14 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o14 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o14 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o14 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o14 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o14 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o14 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o14 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o14 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o14 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o14 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o14 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o14 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o14 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o14 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o14 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o14 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o14 c1o19)
)
)
)
)
(=>
(g1 c1o15)
(and
(=>
(g2 c1o1)
(not
(f c1o15 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o15 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o15 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o15 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o15 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o15 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o15 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o15 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o15 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o15 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o15 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o15 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o15 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o15 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o15 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o15 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o15 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o15 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o15 c1o19)
)
)
)
)
(=>
(g1 c1o16)
(and
(=>
(g2 c1o1)
(not
(f c1o16 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o16 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o16 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o16 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o16 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o16 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o16 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o16 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o16 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o16 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o16 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o16 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o16 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o16 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o16 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o16 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o16 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o16 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o16 c1o19)
)
)
)
)
(=>
(g1 c1o17)
(and
(=>
(g2 c1o1)
(not
(f c1o17 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o17 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o17 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o17 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o17 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o17 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o17 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o17 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o17 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o17 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o17 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o17 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o17 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o17 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o17 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o17 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o17 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o17 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o17 c1o19)
)
)
)
)
(=>
(g1 c1o18)
(and
(=>
(g2 c1o1)
(not
(f c1o18 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o18 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o18 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o18 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o18 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o18 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o18 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o18 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o18 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o18 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o18 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o18 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o18 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o18 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o18 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o18 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o18 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o18 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o18 c1o19)
)
)
)
)
(=>
(g1 c1o19)
(and
(=>
(g2 c1o1)
(not
(f c1o19 c1o1)
)
)
(=>
(g2 c1o2)
(not
(f c1o19 c1o2)
)
)
(=>
(g2 c1o3)
(not
(f c1o19 c1o3)
)
)
(=>
(g2 c1o4)
(not
(f c1o19 c1o4)
)
)
(=>
(g2 c1o5)
(not
(f c1o19 c1o5)
)
)
(=>
(g2 c1o6)
(not
(f c1o19 c1o6)
)
)
(=>
(g2 c1o7)
(not
(f c1o19 c1o7)
)
)
(=>
(g2 c1o8)
(not
(f c1o19 c1o8)
)
)
(=>
(g2 c1o9)
(not
(f c1o19 c1o9)
)
)
(=>
(g2 c1o10)
(not
(f c1o19 c1o10)
)
)
(=>
(g2 c1o11)
(not
(f c1o19 c1o11)
)
)
(=>
(g2 c1o12)
(not
(f c1o19 c1o12)
)
)
(=>
(g2 c1o13)
(not
(f c1o19 c1o13)
)
)
(=>
(g2 c1o14)
(not
(f c1o19 c1o14)
)
)
(=>
(g2 c1o15)
(not
(f c1o19 c1o15)
)
)
(=>
(g2 c1o16)
(not
(f c1o19 c1o16)
)
)
(=>
(g2 c1o17)
(not
(f c1o19 c1o17)
)
)
(=>
(g2 c1o18)
(not
(f c1o19 c1o18)
)
)
(=>
(g2 c1o19)
(not
(f c1o19 c1o19)
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
(g1 c2o3)
(g1 c2o4)
)
(and
(=>
(g1 c2o1)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
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
(=>
(g2 c2o3)
(f c2o1 c2o3)
)
(=>
(g2 c2o4)
(f c2o1 c2o4)
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
(g2 c2o3)
(g2 c2o4)
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
(=>
(g2 c2o3)
(f c2o2 c2o3)
)
(=>
(g2 c2o4)
(f c2o2 c2o4)
)
)
)
)
(=>
(g1 c2o3)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
)
(and
(=>
(g2 c2o1)
(f c2o3 c2o1)
)
(=>
(g2 c2o2)
(f c2o3 c2o2)
)
(=>
(g2 c2o3)
(f c2o3 c2o3)
)
(=>
(g2 c2o4)
(f c2o3 c2o4)
)
)
)
)
(=>
(g1 c2o4)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
)
(and
(=>
(g2 c2o1)
(f c2o4 c2o1)
)
(=>
(g2 c2o2)
(f c2o4 c2o2)
)
(=>
(g2 c2o3)
(f c2o4 c2o3)
)
(=>
(g2 c2o4)
(f c2o4 c2o4)
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
(g1 c2o3)
(g1 c2o4)
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
(and
(g2 c2o3)
(f c2o1 c2o3)
)
(and
(g2 c2o4)
(f c2o1 c2o4)
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
(and
(g2 c2o3)
(f c2o2 c2o3)
)
(and
(g2 c2o4)
(f c2o2 c2o4)
)
)
)
(=>
(g1 c2o3)
(or
(and
(g2 c2o1)
(f c2o3 c2o1)
)
(and
(g2 c2o2)
(f c2o3 c2o2)
)
(and
(g2 c2o3)
(f c2o3 c2o3)
)
(and
(g2 c2o4)
(f c2o3 c2o4)
)
)
)
(=>
(g1 c2o4)
(or
(and
(g2 c2o1)
(f c2o4 c2o1)
)
(and
(g2 c2o2)
(f c2o4 c2o2)
)
(and
(g2 c2o3)
(f c2o4 c2o3)
)
(and
(g2 c2o4)
(f c2o4 c2o4)
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
(g2 c2o3)
(g2 c2o4)
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
(=>
(g2 c2o3)
(f c2o1 c2o3)
)
(=>
(g2 c2o4)
(f c2o1 c2o4)
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
(g2 c2o3)
(g2 c2o4)
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
(=>
(g2 c2o3)
(f c2o2 c2o3)
)
(=>
(g2 c2o4)
(f c2o2 c2o4)
)
)
)
)
(and
(g1 c2o3)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
)
(and
(=>
(g2 c2o1)
(f c2o3 c2o1)
)
(=>
(g2 c2o2)
(f c2o3 c2o2)
)
(=>
(g2 c2o3)
(f c2o3 c2o3)
)
(=>
(g2 c2o4)
(f c2o3 c2o4)
)
)
)
)
(and
(g1 c2o4)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
)
(and
(=>
(g2 c2o1)
(f c2o4 c2o1)
)
(=>
(g2 c2o2)
(f c2o4 c2o2)
)
(=>
(g2 c2o3)
(f c2o4 c2o3)
)
(=>
(g2 c2o4)
(f c2o4 c2o4)
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
(and
(g2 c2o3)
(f c2o1 c2o3)
)
(and
(g2 c2o4)
(f c2o1 c2o4)
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
(and
(g2 c2o3)
(f c2o2 c2o3)
)
(and
(g2 c2o4)
(f c2o2 c2o4)
)
)
)
(and
(g1 c2o3)
(or
(and
(g2 c2o1)
(f c2o3 c2o1)
)
(and
(g2 c2o2)
(f c2o3 c2o2)
)
(and
(g2 c2o3)
(f c2o3 c2o3)
)
(and
(g2 c2o4)
(f c2o3 c2o4)
)
)
)
(and
(g1 c2o4)
(or
(and
(g2 c2o1)
(f c2o4 c2o1)
)
(and
(g2 c2o2)
(f c2o4 c2o2)
)
(and
(g2 c2o3)
(f c2o4 c2o3)
)
(and
(g2 c2o4)
(f c2o4 c2o4)
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
(and
(g2 c2o3)
(not
(f c2o1 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o1 c2o4)
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
(and
(g2 c2o3)
(not
(f c2o2 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o2 c2o4)
)
)
)
)
(and
(g1 c2o3)
(or
(and
(g2 c2o1)
(not
(f c2o3 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o3 c2o2)
)
)
(and
(g2 c2o3)
(not
(f c2o3 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o3 c2o4)
)
)
)
)
(and
(g1 c2o4)
(or
(and
(g2 c2o1)
(not
(f c2o4 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o4 c2o2)
)
)
(and
(g2 c2o3)
(not
(f c2o4 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o4 c2o4)
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
(=>
(g2 c2o3)
(not
(f c2o1 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o1 c2o4)
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
(=>
(g2 c2o3)
(not
(f c2o2 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o2 c2o4)
)
)
)
)
(and
(g1 c2o3)
(and
(=>
(g2 c2o1)
(not
(f c2o3 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o3 c2o2)
)
)
(=>
(g2 c2o3)
(not
(f c2o3 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o3 c2o4)
)
)
)
)
(and
(g1 c2o4)
(and
(=>
(g2 c2o1)
(not
(f c2o4 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o4 c2o2)
)
)
(=>
(g2 c2o3)
(not
(f c2o4 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o4 c2o4)
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
(and
(g2 c2o3)
(not
(f c2o1 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o1 c2o4)
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
(and
(g2 c2o3)
(not
(f c2o2 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o2 c2o4)
)
)
)
)
(=>
(g1 c2o3)
(or
(and
(g2 c2o1)
(not
(f c2o3 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o3 c2o2)
)
)
(and
(g2 c2o3)
(not
(f c2o3 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o3 c2o4)
)
)
)
)
(=>
(g1 c2o4)
(or
(and
(g2 c2o1)
(not
(f c2o4 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o4 c2o2)
)
)
(and
(g2 c2o3)
(not
(f c2o4 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o4 c2o4)
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
(=>
(g2 c2o3)
(not
(f c2o1 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o1 c2o4)
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
(=>
(g2 c2o3)
(not
(f c2o2 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o2 c2o4)
)
)
)
)
(=>
(g1 c2o3)
(and
(=>
(g2 c2o1)
(not
(f c2o3 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o3 c2o2)
)
)
(=>
(g2 c2o3)
(not
(f c2o3 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o3 c2o4)
)
)
)
)
(=>
(g1 c2o4)
(and
(=>
(g2 c2o1)
(not
(f c2o4 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o4 c2o2)
)
)
(=>
(g2 c2o3)
(not
(f c2o4 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o4 c2o4)
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
)
(and
(=>
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
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
)
)
)
)
)
)
)

(assert
(=>
(= c4 true)
(and
(=>
(and
(= q1 true)
(= q2 true)
)
(and
(or
(g1 c4o1)
(g1 c4o2)
(g1 c4o3)
(g1 c4o4)
(g1 c4o5)
(g1 c4o6)
(g1 c4o7)
(g1 c4o8)
(g1 c4o9)
(g1 c4o10)
(g1 c4o11)
(g1 c4o12)
)
(and
(=>
(g1 c4o1)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o1 c4o1)
)
(=>
(g2 c4o2)
(f c4o1 c4o2)
)
(=>
(g2 c4o3)
(f c4o1 c4o3)
)
(=>
(g2 c4o4)
(f c4o1 c4o4)
)
(=>
(g2 c4o5)
(f c4o1 c4o5)
)
(=>
(g2 c4o6)
(f c4o1 c4o6)
)
(=>
(g2 c4o7)
(f c4o1 c4o7)
)
(=>
(g2 c4o8)
(f c4o1 c4o8)
)
(=>
(g2 c4o9)
(f c4o1 c4o9)
)
(=>
(g2 c4o10)
(f c4o1 c4o10)
)
(=>
(g2 c4o11)
(f c4o1 c4o11)
)
(=>
(g2 c4o12)
(f c4o1 c4o12)
)
)
)
)
(=>
(g1 c4o2)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o2 c4o1)
)
(=>
(g2 c4o2)
(f c4o2 c4o2)
)
(=>
(g2 c4o3)
(f c4o2 c4o3)
)
(=>
(g2 c4o4)
(f c4o2 c4o4)
)
(=>
(g2 c4o5)
(f c4o2 c4o5)
)
(=>
(g2 c4o6)
(f c4o2 c4o6)
)
(=>
(g2 c4o7)
(f c4o2 c4o7)
)
(=>
(g2 c4o8)
(f c4o2 c4o8)
)
(=>
(g2 c4o9)
(f c4o2 c4o9)
)
(=>
(g2 c4o10)
(f c4o2 c4o10)
)
(=>
(g2 c4o11)
(f c4o2 c4o11)
)
(=>
(g2 c4o12)
(f c4o2 c4o12)
)
)
)
)
(=>
(g1 c4o3)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o3 c4o1)
)
(=>
(g2 c4o2)
(f c4o3 c4o2)
)
(=>
(g2 c4o3)
(f c4o3 c4o3)
)
(=>
(g2 c4o4)
(f c4o3 c4o4)
)
(=>
(g2 c4o5)
(f c4o3 c4o5)
)
(=>
(g2 c4o6)
(f c4o3 c4o6)
)
(=>
(g2 c4o7)
(f c4o3 c4o7)
)
(=>
(g2 c4o8)
(f c4o3 c4o8)
)
(=>
(g2 c4o9)
(f c4o3 c4o9)
)
(=>
(g2 c4o10)
(f c4o3 c4o10)
)
(=>
(g2 c4o11)
(f c4o3 c4o11)
)
(=>
(g2 c4o12)
(f c4o3 c4o12)
)
)
)
)
(=>
(g1 c4o4)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o4 c4o1)
)
(=>
(g2 c4o2)
(f c4o4 c4o2)
)
(=>
(g2 c4o3)
(f c4o4 c4o3)
)
(=>
(g2 c4o4)
(f c4o4 c4o4)
)
(=>
(g2 c4o5)
(f c4o4 c4o5)
)
(=>
(g2 c4o6)
(f c4o4 c4o6)
)
(=>
(g2 c4o7)
(f c4o4 c4o7)
)
(=>
(g2 c4o8)
(f c4o4 c4o8)
)
(=>
(g2 c4o9)
(f c4o4 c4o9)
)
(=>
(g2 c4o10)
(f c4o4 c4o10)
)
(=>
(g2 c4o11)
(f c4o4 c4o11)
)
(=>
(g2 c4o12)
(f c4o4 c4o12)
)
)
)
)
(=>
(g1 c4o5)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o5 c4o1)
)
(=>
(g2 c4o2)
(f c4o5 c4o2)
)
(=>
(g2 c4o3)
(f c4o5 c4o3)
)
(=>
(g2 c4o4)
(f c4o5 c4o4)
)
(=>
(g2 c4o5)
(f c4o5 c4o5)
)
(=>
(g2 c4o6)
(f c4o5 c4o6)
)
(=>
(g2 c4o7)
(f c4o5 c4o7)
)
(=>
(g2 c4o8)
(f c4o5 c4o8)
)
(=>
(g2 c4o9)
(f c4o5 c4o9)
)
(=>
(g2 c4o10)
(f c4o5 c4o10)
)
(=>
(g2 c4o11)
(f c4o5 c4o11)
)
(=>
(g2 c4o12)
(f c4o5 c4o12)
)
)
)
)
(=>
(g1 c4o6)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o6 c4o1)
)
(=>
(g2 c4o2)
(f c4o6 c4o2)
)
(=>
(g2 c4o3)
(f c4o6 c4o3)
)
(=>
(g2 c4o4)
(f c4o6 c4o4)
)
(=>
(g2 c4o5)
(f c4o6 c4o5)
)
(=>
(g2 c4o6)
(f c4o6 c4o6)
)
(=>
(g2 c4o7)
(f c4o6 c4o7)
)
(=>
(g2 c4o8)
(f c4o6 c4o8)
)
(=>
(g2 c4o9)
(f c4o6 c4o9)
)
(=>
(g2 c4o10)
(f c4o6 c4o10)
)
(=>
(g2 c4o11)
(f c4o6 c4o11)
)
(=>
(g2 c4o12)
(f c4o6 c4o12)
)
)
)
)
(=>
(g1 c4o7)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o7 c4o1)
)
(=>
(g2 c4o2)
(f c4o7 c4o2)
)
(=>
(g2 c4o3)
(f c4o7 c4o3)
)
(=>
(g2 c4o4)
(f c4o7 c4o4)
)
(=>
(g2 c4o5)
(f c4o7 c4o5)
)
(=>
(g2 c4o6)
(f c4o7 c4o6)
)
(=>
(g2 c4o7)
(f c4o7 c4o7)
)
(=>
(g2 c4o8)
(f c4o7 c4o8)
)
(=>
(g2 c4o9)
(f c4o7 c4o9)
)
(=>
(g2 c4o10)
(f c4o7 c4o10)
)
(=>
(g2 c4o11)
(f c4o7 c4o11)
)
(=>
(g2 c4o12)
(f c4o7 c4o12)
)
)
)
)
(=>
(g1 c4o8)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o8 c4o1)
)
(=>
(g2 c4o2)
(f c4o8 c4o2)
)
(=>
(g2 c4o3)
(f c4o8 c4o3)
)
(=>
(g2 c4o4)
(f c4o8 c4o4)
)
(=>
(g2 c4o5)
(f c4o8 c4o5)
)
(=>
(g2 c4o6)
(f c4o8 c4o6)
)
(=>
(g2 c4o7)
(f c4o8 c4o7)
)
(=>
(g2 c4o8)
(f c4o8 c4o8)
)
(=>
(g2 c4o9)
(f c4o8 c4o9)
)
(=>
(g2 c4o10)
(f c4o8 c4o10)
)
(=>
(g2 c4o11)
(f c4o8 c4o11)
)
(=>
(g2 c4o12)
(f c4o8 c4o12)
)
)
)
)
(=>
(g1 c4o9)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o9 c4o1)
)
(=>
(g2 c4o2)
(f c4o9 c4o2)
)
(=>
(g2 c4o3)
(f c4o9 c4o3)
)
(=>
(g2 c4o4)
(f c4o9 c4o4)
)
(=>
(g2 c4o5)
(f c4o9 c4o5)
)
(=>
(g2 c4o6)
(f c4o9 c4o6)
)
(=>
(g2 c4o7)
(f c4o9 c4o7)
)
(=>
(g2 c4o8)
(f c4o9 c4o8)
)
(=>
(g2 c4o9)
(f c4o9 c4o9)
)
(=>
(g2 c4o10)
(f c4o9 c4o10)
)
(=>
(g2 c4o11)
(f c4o9 c4o11)
)
(=>
(g2 c4o12)
(f c4o9 c4o12)
)
)
)
)
(=>
(g1 c4o10)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o10 c4o1)
)
(=>
(g2 c4o2)
(f c4o10 c4o2)
)
(=>
(g2 c4o3)
(f c4o10 c4o3)
)
(=>
(g2 c4o4)
(f c4o10 c4o4)
)
(=>
(g2 c4o5)
(f c4o10 c4o5)
)
(=>
(g2 c4o6)
(f c4o10 c4o6)
)
(=>
(g2 c4o7)
(f c4o10 c4o7)
)
(=>
(g2 c4o8)
(f c4o10 c4o8)
)
(=>
(g2 c4o9)
(f c4o10 c4o9)
)
(=>
(g2 c4o10)
(f c4o10 c4o10)
)
(=>
(g2 c4o11)
(f c4o10 c4o11)
)
(=>
(g2 c4o12)
(f c4o10 c4o12)
)
)
)
)
(=>
(g1 c4o11)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o11 c4o1)
)
(=>
(g2 c4o2)
(f c4o11 c4o2)
)
(=>
(g2 c4o3)
(f c4o11 c4o3)
)
(=>
(g2 c4o4)
(f c4o11 c4o4)
)
(=>
(g2 c4o5)
(f c4o11 c4o5)
)
(=>
(g2 c4o6)
(f c4o11 c4o6)
)
(=>
(g2 c4o7)
(f c4o11 c4o7)
)
(=>
(g2 c4o8)
(f c4o11 c4o8)
)
(=>
(g2 c4o9)
(f c4o11 c4o9)
)
(=>
(g2 c4o10)
(f c4o11 c4o10)
)
(=>
(g2 c4o11)
(f c4o11 c4o11)
)
(=>
(g2 c4o12)
(f c4o11 c4o12)
)
)
)
)
(=>
(g1 c4o12)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o12 c4o1)
)
(=>
(g2 c4o2)
(f c4o12 c4o2)
)
(=>
(g2 c4o3)
(f c4o12 c4o3)
)
(=>
(g2 c4o4)
(f c4o12 c4o4)
)
(=>
(g2 c4o5)
(f c4o12 c4o5)
)
(=>
(g2 c4o6)
(f c4o12 c4o6)
)
(=>
(g2 c4o7)
(f c4o12 c4o7)
)
(=>
(g2 c4o8)
(f c4o12 c4o8)
)
(=>
(g2 c4o9)
(f c4o12 c4o9)
)
(=>
(g2 c4o10)
(f c4o12 c4o10)
)
(=>
(g2 c4o11)
(f c4o12 c4o11)
)
(=>
(g2 c4o12)
(f c4o12 c4o12)
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
(g1 c4o1)
(g1 c4o2)
(g1 c4o3)
(g1 c4o4)
(g1 c4o5)
(g1 c4o6)
(g1 c4o7)
(g1 c4o8)
(g1 c4o9)
(g1 c4o10)
(g1 c4o11)
(g1 c4o12)
)
(and
(=>
(g1 c4o1)
(or
(and
(g2 c4o1)
(f c4o1 c4o1)
)
(and
(g2 c4o2)
(f c4o1 c4o2)
)
(and
(g2 c4o3)
(f c4o1 c4o3)
)
(and
(g2 c4o4)
(f c4o1 c4o4)
)
(and
(g2 c4o5)
(f c4o1 c4o5)
)
(and
(g2 c4o6)
(f c4o1 c4o6)
)
(and
(g2 c4o7)
(f c4o1 c4o7)
)
(and
(g2 c4o8)
(f c4o1 c4o8)
)
(and
(g2 c4o9)
(f c4o1 c4o9)
)
(and
(g2 c4o10)
(f c4o1 c4o10)
)
(and
(g2 c4o11)
(f c4o1 c4o11)
)
(and
(g2 c4o12)
(f c4o1 c4o12)
)
)
)
(=>
(g1 c4o2)
(or
(and
(g2 c4o1)
(f c4o2 c4o1)
)
(and
(g2 c4o2)
(f c4o2 c4o2)
)
(and
(g2 c4o3)
(f c4o2 c4o3)
)
(and
(g2 c4o4)
(f c4o2 c4o4)
)
(and
(g2 c4o5)
(f c4o2 c4o5)
)
(and
(g2 c4o6)
(f c4o2 c4o6)
)
(and
(g2 c4o7)
(f c4o2 c4o7)
)
(and
(g2 c4o8)
(f c4o2 c4o8)
)
(and
(g2 c4o9)
(f c4o2 c4o9)
)
(and
(g2 c4o10)
(f c4o2 c4o10)
)
(and
(g2 c4o11)
(f c4o2 c4o11)
)
(and
(g2 c4o12)
(f c4o2 c4o12)
)
)
)
(=>
(g1 c4o3)
(or
(and
(g2 c4o1)
(f c4o3 c4o1)
)
(and
(g2 c4o2)
(f c4o3 c4o2)
)
(and
(g2 c4o3)
(f c4o3 c4o3)
)
(and
(g2 c4o4)
(f c4o3 c4o4)
)
(and
(g2 c4o5)
(f c4o3 c4o5)
)
(and
(g2 c4o6)
(f c4o3 c4o6)
)
(and
(g2 c4o7)
(f c4o3 c4o7)
)
(and
(g2 c4o8)
(f c4o3 c4o8)
)
(and
(g2 c4o9)
(f c4o3 c4o9)
)
(and
(g2 c4o10)
(f c4o3 c4o10)
)
(and
(g2 c4o11)
(f c4o3 c4o11)
)
(and
(g2 c4o12)
(f c4o3 c4o12)
)
)
)
(=>
(g1 c4o4)
(or
(and
(g2 c4o1)
(f c4o4 c4o1)
)
(and
(g2 c4o2)
(f c4o4 c4o2)
)
(and
(g2 c4o3)
(f c4o4 c4o3)
)
(and
(g2 c4o4)
(f c4o4 c4o4)
)
(and
(g2 c4o5)
(f c4o4 c4o5)
)
(and
(g2 c4o6)
(f c4o4 c4o6)
)
(and
(g2 c4o7)
(f c4o4 c4o7)
)
(and
(g2 c4o8)
(f c4o4 c4o8)
)
(and
(g2 c4o9)
(f c4o4 c4o9)
)
(and
(g2 c4o10)
(f c4o4 c4o10)
)
(and
(g2 c4o11)
(f c4o4 c4o11)
)
(and
(g2 c4o12)
(f c4o4 c4o12)
)
)
)
(=>
(g1 c4o5)
(or
(and
(g2 c4o1)
(f c4o5 c4o1)
)
(and
(g2 c4o2)
(f c4o5 c4o2)
)
(and
(g2 c4o3)
(f c4o5 c4o3)
)
(and
(g2 c4o4)
(f c4o5 c4o4)
)
(and
(g2 c4o5)
(f c4o5 c4o5)
)
(and
(g2 c4o6)
(f c4o5 c4o6)
)
(and
(g2 c4o7)
(f c4o5 c4o7)
)
(and
(g2 c4o8)
(f c4o5 c4o8)
)
(and
(g2 c4o9)
(f c4o5 c4o9)
)
(and
(g2 c4o10)
(f c4o5 c4o10)
)
(and
(g2 c4o11)
(f c4o5 c4o11)
)
(and
(g2 c4o12)
(f c4o5 c4o12)
)
)
)
(=>
(g1 c4o6)
(or
(and
(g2 c4o1)
(f c4o6 c4o1)
)
(and
(g2 c4o2)
(f c4o6 c4o2)
)
(and
(g2 c4o3)
(f c4o6 c4o3)
)
(and
(g2 c4o4)
(f c4o6 c4o4)
)
(and
(g2 c4o5)
(f c4o6 c4o5)
)
(and
(g2 c4o6)
(f c4o6 c4o6)
)
(and
(g2 c4o7)
(f c4o6 c4o7)
)
(and
(g2 c4o8)
(f c4o6 c4o8)
)
(and
(g2 c4o9)
(f c4o6 c4o9)
)
(and
(g2 c4o10)
(f c4o6 c4o10)
)
(and
(g2 c4o11)
(f c4o6 c4o11)
)
(and
(g2 c4o12)
(f c4o6 c4o12)
)
)
)
(=>
(g1 c4o7)
(or
(and
(g2 c4o1)
(f c4o7 c4o1)
)
(and
(g2 c4o2)
(f c4o7 c4o2)
)
(and
(g2 c4o3)
(f c4o7 c4o3)
)
(and
(g2 c4o4)
(f c4o7 c4o4)
)
(and
(g2 c4o5)
(f c4o7 c4o5)
)
(and
(g2 c4o6)
(f c4o7 c4o6)
)
(and
(g2 c4o7)
(f c4o7 c4o7)
)
(and
(g2 c4o8)
(f c4o7 c4o8)
)
(and
(g2 c4o9)
(f c4o7 c4o9)
)
(and
(g2 c4o10)
(f c4o7 c4o10)
)
(and
(g2 c4o11)
(f c4o7 c4o11)
)
(and
(g2 c4o12)
(f c4o7 c4o12)
)
)
)
(=>
(g1 c4o8)
(or
(and
(g2 c4o1)
(f c4o8 c4o1)
)
(and
(g2 c4o2)
(f c4o8 c4o2)
)
(and
(g2 c4o3)
(f c4o8 c4o3)
)
(and
(g2 c4o4)
(f c4o8 c4o4)
)
(and
(g2 c4o5)
(f c4o8 c4o5)
)
(and
(g2 c4o6)
(f c4o8 c4o6)
)
(and
(g2 c4o7)
(f c4o8 c4o7)
)
(and
(g2 c4o8)
(f c4o8 c4o8)
)
(and
(g2 c4o9)
(f c4o8 c4o9)
)
(and
(g2 c4o10)
(f c4o8 c4o10)
)
(and
(g2 c4o11)
(f c4o8 c4o11)
)
(and
(g2 c4o12)
(f c4o8 c4o12)
)
)
)
(=>
(g1 c4o9)
(or
(and
(g2 c4o1)
(f c4o9 c4o1)
)
(and
(g2 c4o2)
(f c4o9 c4o2)
)
(and
(g2 c4o3)
(f c4o9 c4o3)
)
(and
(g2 c4o4)
(f c4o9 c4o4)
)
(and
(g2 c4o5)
(f c4o9 c4o5)
)
(and
(g2 c4o6)
(f c4o9 c4o6)
)
(and
(g2 c4o7)
(f c4o9 c4o7)
)
(and
(g2 c4o8)
(f c4o9 c4o8)
)
(and
(g2 c4o9)
(f c4o9 c4o9)
)
(and
(g2 c4o10)
(f c4o9 c4o10)
)
(and
(g2 c4o11)
(f c4o9 c4o11)
)
(and
(g2 c4o12)
(f c4o9 c4o12)
)
)
)
(=>
(g1 c4o10)
(or
(and
(g2 c4o1)
(f c4o10 c4o1)
)
(and
(g2 c4o2)
(f c4o10 c4o2)
)
(and
(g2 c4o3)
(f c4o10 c4o3)
)
(and
(g2 c4o4)
(f c4o10 c4o4)
)
(and
(g2 c4o5)
(f c4o10 c4o5)
)
(and
(g2 c4o6)
(f c4o10 c4o6)
)
(and
(g2 c4o7)
(f c4o10 c4o7)
)
(and
(g2 c4o8)
(f c4o10 c4o8)
)
(and
(g2 c4o9)
(f c4o10 c4o9)
)
(and
(g2 c4o10)
(f c4o10 c4o10)
)
(and
(g2 c4o11)
(f c4o10 c4o11)
)
(and
(g2 c4o12)
(f c4o10 c4o12)
)
)
)
(=>
(g1 c4o11)
(or
(and
(g2 c4o1)
(f c4o11 c4o1)
)
(and
(g2 c4o2)
(f c4o11 c4o2)
)
(and
(g2 c4o3)
(f c4o11 c4o3)
)
(and
(g2 c4o4)
(f c4o11 c4o4)
)
(and
(g2 c4o5)
(f c4o11 c4o5)
)
(and
(g2 c4o6)
(f c4o11 c4o6)
)
(and
(g2 c4o7)
(f c4o11 c4o7)
)
(and
(g2 c4o8)
(f c4o11 c4o8)
)
(and
(g2 c4o9)
(f c4o11 c4o9)
)
(and
(g2 c4o10)
(f c4o11 c4o10)
)
(and
(g2 c4o11)
(f c4o11 c4o11)
)
(and
(g2 c4o12)
(f c4o11 c4o12)
)
)
)
(=>
(g1 c4o12)
(or
(and
(g2 c4o1)
(f c4o12 c4o1)
)
(and
(g2 c4o2)
(f c4o12 c4o2)
)
(and
(g2 c4o3)
(f c4o12 c4o3)
)
(and
(g2 c4o4)
(f c4o12 c4o4)
)
(and
(g2 c4o5)
(f c4o12 c4o5)
)
(and
(g2 c4o6)
(f c4o12 c4o6)
)
(and
(g2 c4o7)
(f c4o12 c4o7)
)
(and
(g2 c4o8)
(f c4o12 c4o8)
)
(and
(g2 c4o9)
(f c4o12 c4o9)
)
(and
(g2 c4o10)
(f c4o12 c4o10)
)
(and
(g2 c4o11)
(f c4o12 c4o11)
)
(and
(g2 c4o12)
(f c4o12 c4o12)
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
(g1 c4o1)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o1 c4o1)
)
(=>
(g2 c4o2)
(f c4o1 c4o2)
)
(=>
(g2 c4o3)
(f c4o1 c4o3)
)
(=>
(g2 c4o4)
(f c4o1 c4o4)
)
(=>
(g2 c4o5)
(f c4o1 c4o5)
)
(=>
(g2 c4o6)
(f c4o1 c4o6)
)
(=>
(g2 c4o7)
(f c4o1 c4o7)
)
(=>
(g2 c4o8)
(f c4o1 c4o8)
)
(=>
(g2 c4o9)
(f c4o1 c4o9)
)
(=>
(g2 c4o10)
(f c4o1 c4o10)
)
(=>
(g2 c4o11)
(f c4o1 c4o11)
)
(=>
(g2 c4o12)
(f c4o1 c4o12)
)
)
)
)
(and
(g1 c4o2)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o2 c4o1)
)
(=>
(g2 c4o2)
(f c4o2 c4o2)
)
(=>
(g2 c4o3)
(f c4o2 c4o3)
)
(=>
(g2 c4o4)
(f c4o2 c4o4)
)
(=>
(g2 c4o5)
(f c4o2 c4o5)
)
(=>
(g2 c4o6)
(f c4o2 c4o6)
)
(=>
(g2 c4o7)
(f c4o2 c4o7)
)
(=>
(g2 c4o8)
(f c4o2 c4o8)
)
(=>
(g2 c4o9)
(f c4o2 c4o9)
)
(=>
(g2 c4o10)
(f c4o2 c4o10)
)
(=>
(g2 c4o11)
(f c4o2 c4o11)
)
(=>
(g2 c4o12)
(f c4o2 c4o12)
)
)
)
)
(and
(g1 c4o3)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o3 c4o1)
)
(=>
(g2 c4o2)
(f c4o3 c4o2)
)
(=>
(g2 c4o3)
(f c4o3 c4o3)
)
(=>
(g2 c4o4)
(f c4o3 c4o4)
)
(=>
(g2 c4o5)
(f c4o3 c4o5)
)
(=>
(g2 c4o6)
(f c4o3 c4o6)
)
(=>
(g2 c4o7)
(f c4o3 c4o7)
)
(=>
(g2 c4o8)
(f c4o3 c4o8)
)
(=>
(g2 c4o9)
(f c4o3 c4o9)
)
(=>
(g2 c4o10)
(f c4o3 c4o10)
)
(=>
(g2 c4o11)
(f c4o3 c4o11)
)
(=>
(g2 c4o12)
(f c4o3 c4o12)
)
)
)
)
(and
(g1 c4o4)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o4 c4o1)
)
(=>
(g2 c4o2)
(f c4o4 c4o2)
)
(=>
(g2 c4o3)
(f c4o4 c4o3)
)
(=>
(g2 c4o4)
(f c4o4 c4o4)
)
(=>
(g2 c4o5)
(f c4o4 c4o5)
)
(=>
(g2 c4o6)
(f c4o4 c4o6)
)
(=>
(g2 c4o7)
(f c4o4 c4o7)
)
(=>
(g2 c4o8)
(f c4o4 c4o8)
)
(=>
(g2 c4o9)
(f c4o4 c4o9)
)
(=>
(g2 c4o10)
(f c4o4 c4o10)
)
(=>
(g2 c4o11)
(f c4o4 c4o11)
)
(=>
(g2 c4o12)
(f c4o4 c4o12)
)
)
)
)
(and
(g1 c4o5)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o5 c4o1)
)
(=>
(g2 c4o2)
(f c4o5 c4o2)
)
(=>
(g2 c4o3)
(f c4o5 c4o3)
)
(=>
(g2 c4o4)
(f c4o5 c4o4)
)
(=>
(g2 c4o5)
(f c4o5 c4o5)
)
(=>
(g2 c4o6)
(f c4o5 c4o6)
)
(=>
(g2 c4o7)
(f c4o5 c4o7)
)
(=>
(g2 c4o8)
(f c4o5 c4o8)
)
(=>
(g2 c4o9)
(f c4o5 c4o9)
)
(=>
(g2 c4o10)
(f c4o5 c4o10)
)
(=>
(g2 c4o11)
(f c4o5 c4o11)
)
(=>
(g2 c4o12)
(f c4o5 c4o12)
)
)
)
)
(and
(g1 c4o6)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o6 c4o1)
)
(=>
(g2 c4o2)
(f c4o6 c4o2)
)
(=>
(g2 c4o3)
(f c4o6 c4o3)
)
(=>
(g2 c4o4)
(f c4o6 c4o4)
)
(=>
(g2 c4o5)
(f c4o6 c4o5)
)
(=>
(g2 c4o6)
(f c4o6 c4o6)
)
(=>
(g2 c4o7)
(f c4o6 c4o7)
)
(=>
(g2 c4o8)
(f c4o6 c4o8)
)
(=>
(g2 c4o9)
(f c4o6 c4o9)
)
(=>
(g2 c4o10)
(f c4o6 c4o10)
)
(=>
(g2 c4o11)
(f c4o6 c4o11)
)
(=>
(g2 c4o12)
(f c4o6 c4o12)
)
)
)
)
(and
(g1 c4o7)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o7 c4o1)
)
(=>
(g2 c4o2)
(f c4o7 c4o2)
)
(=>
(g2 c4o3)
(f c4o7 c4o3)
)
(=>
(g2 c4o4)
(f c4o7 c4o4)
)
(=>
(g2 c4o5)
(f c4o7 c4o5)
)
(=>
(g2 c4o6)
(f c4o7 c4o6)
)
(=>
(g2 c4o7)
(f c4o7 c4o7)
)
(=>
(g2 c4o8)
(f c4o7 c4o8)
)
(=>
(g2 c4o9)
(f c4o7 c4o9)
)
(=>
(g2 c4o10)
(f c4o7 c4o10)
)
(=>
(g2 c4o11)
(f c4o7 c4o11)
)
(=>
(g2 c4o12)
(f c4o7 c4o12)
)
)
)
)
(and
(g1 c4o8)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o8 c4o1)
)
(=>
(g2 c4o2)
(f c4o8 c4o2)
)
(=>
(g2 c4o3)
(f c4o8 c4o3)
)
(=>
(g2 c4o4)
(f c4o8 c4o4)
)
(=>
(g2 c4o5)
(f c4o8 c4o5)
)
(=>
(g2 c4o6)
(f c4o8 c4o6)
)
(=>
(g2 c4o7)
(f c4o8 c4o7)
)
(=>
(g2 c4o8)
(f c4o8 c4o8)
)
(=>
(g2 c4o9)
(f c4o8 c4o9)
)
(=>
(g2 c4o10)
(f c4o8 c4o10)
)
(=>
(g2 c4o11)
(f c4o8 c4o11)
)
(=>
(g2 c4o12)
(f c4o8 c4o12)
)
)
)
)
(and
(g1 c4o9)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o9 c4o1)
)
(=>
(g2 c4o2)
(f c4o9 c4o2)
)
(=>
(g2 c4o3)
(f c4o9 c4o3)
)
(=>
(g2 c4o4)
(f c4o9 c4o4)
)
(=>
(g2 c4o5)
(f c4o9 c4o5)
)
(=>
(g2 c4o6)
(f c4o9 c4o6)
)
(=>
(g2 c4o7)
(f c4o9 c4o7)
)
(=>
(g2 c4o8)
(f c4o9 c4o8)
)
(=>
(g2 c4o9)
(f c4o9 c4o9)
)
(=>
(g2 c4o10)
(f c4o9 c4o10)
)
(=>
(g2 c4o11)
(f c4o9 c4o11)
)
(=>
(g2 c4o12)
(f c4o9 c4o12)
)
)
)
)
(and
(g1 c4o10)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o10 c4o1)
)
(=>
(g2 c4o2)
(f c4o10 c4o2)
)
(=>
(g2 c4o3)
(f c4o10 c4o3)
)
(=>
(g2 c4o4)
(f c4o10 c4o4)
)
(=>
(g2 c4o5)
(f c4o10 c4o5)
)
(=>
(g2 c4o6)
(f c4o10 c4o6)
)
(=>
(g2 c4o7)
(f c4o10 c4o7)
)
(=>
(g2 c4o8)
(f c4o10 c4o8)
)
(=>
(g2 c4o9)
(f c4o10 c4o9)
)
(=>
(g2 c4o10)
(f c4o10 c4o10)
)
(=>
(g2 c4o11)
(f c4o10 c4o11)
)
(=>
(g2 c4o12)
(f c4o10 c4o12)
)
)
)
)
(and
(g1 c4o11)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o11 c4o1)
)
(=>
(g2 c4o2)
(f c4o11 c4o2)
)
(=>
(g2 c4o3)
(f c4o11 c4o3)
)
(=>
(g2 c4o4)
(f c4o11 c4o4)
)
(=>
(g2 c4o5)
(f c4o11 c4o5)
)
(=>
(g2 c4o6)
(f c4o11 c4o6)
)
(=>
(g2 c4o7)
(f c4o11 c4o7)
)
(=>
(g2 c4o8)
(f c4o11 c4o8)
)
(=>
(g2 c4o9)
(f c4o11 c4o9)
)
(=>
(g2 c4o10)
(f c4o11 c4o10)
)
(=>
(g2 c4o11)
(f c4o11 c4o11)
)
(=>
(g2 c4o12)
(f c4o11 c4o12)
)
)
)
)
(and
(g1 c4o12)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
(g2 c4o4)
(g2 c4o5)
(g2 c4o6)
(g2 c4o7)
(g2 c4o8)
(g2 c4o9)
(g2 c4o10)
(g2 c4o11)
(g2 c4o12)
)
(and
(=>
(g2 c4o1)
(f c4o12 c4o1)
)
(=>
(g2 c4o2)
(f c4o12 c4o2)
)
(=>
(g2 c4o3)
(f c4o12 c4o3)
)
(=>
(g2 c4o4)
(f c4o12 c4o4)
)
(=>
(g2 c4o5)
(f c4o12 c4o5)
)
(=>
(g2 c4o6)
(f c4o12 c4o6)
)
(=>
(g2 c4o7)
(f c4o12 c4o7)
)
(=>
(g2 c4o8)
(f c4o12 c4o8)
)
(=>
(g2 c4o9)
(f c4o12 c4o9)
)
(=>
(g2 c4o10)
(f c4o12 c4o10)
)
(=>
(g2 c4o11)
(f c4o12 c4o11)
)
(=>
(g2 c4o12)
(f c4o12 c4o12)
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
(g1 c4o1)
(or
(and
(g2 c4o1)
(f c4o1 c4o1)
)
(and
(g2 c4o2)
(f c4o1 c4o2)
)
(and
(g2 c4o3)
(f c4o1 c4o3)
)
(and
(g2 c4o4)
(f c4o1 c4o4)
)
(and
(g2 c4o5)
(f c4o1 c4o5)
)
(and
(g2 c4o6)
(f c4o1 c4o6)
)
(and
(g2 c4o7)
(f c4o1 c4o7)
)
(and
(g2 c4o8)
(f c4o1 c4o8)
)
(and
(g2 c4o9)
(f c4o1 c4o9)
)
(and
(g2 c4o10)
(f c4o1 c4o10)
)
(and
(g2 c4o11)
(f c4o1 c4o11)
)
(and
(g2 c4o12)
(f c4o1 c4o12)
)
)
)
(and
(g1 c4o2)
(or
(and
(g2 c4o1)
(f c4o2 c4o1)
)
(and
(g2 c4o2)
(f c4o2 c4o2)
)
(and
(g2 c4o3)
(f c4o2 c4o3)
)
(and
(g2 c4o4)
(f c4o2 c4o4)
)
(and
(g2 c4o5)
(f c4o2 c4o5)
)
(and
(g2 c4o6)
(f c4o2 c4o6)
)
(and
(g2 c4o7)
(f c4o2 c4o7)
)
(and
(g2 c4o8)
(f c4o2 c4o8)
)
(and
(g2 c4o9)
(f c4o2 c4o9)
)
(and
(g2 c4o10)
(f c4o2 c4o10)
)
(and
(g2 c4o11)
(f c4o2 c4o11)
)
(and
(g2 c4o12)
(f c4o2 c4o12)
)
)
)
(and
(g1 c4o3)
(or
(and
(g2 c4o1)
(f c4o3 c4o1)
)
(and
(g2 c4o2)
(f c4o3 c4o2)
)
(and
(g2 c4o3)
(f c4o3 c4o3)
)
(and
(g2 c4o4)
(f c4o3 c4o4)
)
(and
(g2 c4o5)
(f c4o3 c4o5)
)
(and
(g2 c4o6)
(f c4o3 c4o6)
)
(and
(g2 c4o7)
(f c4o3 c4o7)
)
(and
(g2 c4o8)
(f c4o3 c4o8)
)
(and
(g2 c4o9)
(f c4o3 c4o9)
)
(and
(g2 c4o10)
(f c4o3 c4o10)
)
(and
(g2 c4o11)
(f c4o3 c4o11)
)
(and
(g2 c4o12)
(f c4o3 c4o12)
)
)
)
(and
(g1 c4o4)
(or
(and
(g2 c4o1)
(f c4o4 c4o1)
)
(and
(g2 c4o2)
(f c4o4 c4o2)
)
(and
(g2 c4o3)
(f c4o4 c4o3)
)
(and
(g2 c4o4)
(f c4o4 c4o4)
)
(and
(g2 c4o5)
(f c4o4 c4o5)
)
(and
(g2 c4o6)
(f c4o4 c4o6)
)
(and
(g2 c4o7)
(f c4o4 c4o7)
)
(and
(g2 c4o8)
(f c4o4 c4o8)
)
(and
(g2 c4o9)
(f c4o4 c4o9)
)
(and
(g2 c4o10)
(f c4o4 c4o10)
)
(and
(g2 c4o11)
(f c4o4 c4o11)
)
(and
(g2 c4o12)
(f c4o4 c4o12)
)
)
)
(and
(g1 c4o5)
(or
(and
(g2 c4o1)
(f c4o5 c4o1)
)
(and
(g2 c4o2)
(f c4o5 c4o2)
)
(and
(g2 c4o3)
(f c4o5 c4o3)
)
(and
(g2 c4o4)
(f c4o5 c4o4)
)
(and
(g2 c4o5)
(f c4o5 c4o5)
)
(and
(g2 c4o6)
(f c4o5 c4o6)
)
(and
(g2 c4o7)
(f c4o5 c4o7)
)
(and
(g2 c4o8)
(f c4o5 c4o8)
)
(and
(g2 c4o9)
(f c4o5 c4o9)
)
(and
(g2 c4o10)
(f c4o5 c4o10)
)
(and
(g2 c4o11)
(f c4o5 c4o11)
)
(and
(g2 c4o12)
(f c4o5 c4o12)
)
)
)
(and
(g1 c4o6)
(or
(and
(g2 c4o1)
(f c4o6 c4o1)
)
(and
(g2 c4o2)
(f c4o6 c4o2)
)
(and
(g2 c4o3)
(f c4o6 c4o3)
)
(and
(g2 c4o4)
(f c4o6 c4o4)
)
(and
(g2 c4o5)
(f c4o6 c4o5)
)
(and
(g2 c4o6)
(f c4o6 c4o6)
)
(and
(g2 c4o7)
(f c4o6 c4o7)
)
(and
(g2 c4o8)
(f c4o6 c4o8)
)
(and
(g2 c4o9)
(f c4o6 c4o9)
)
(and
(g2 c4o10)
(f c4o6 c4o10)
)
(and
(g2 c4o11)
(f c4o6 c4o11)
)
(and
(g2 c4o12)
(f c4o6 c4o12)
)
)
)
(and
(g1 c4o7)
(or
(and
(g2 c4o1)
(f c4o7 c4o1)
)
(and
(g2 c4o2)
(f c4o7 c4o2)
)
(and
(g2 c4o3)
(f c4o7 c4o3)
)
(and
(g2 c4o4)
(f c4o7 c4o4)
)
(and
(g2 c4o5)
(f c4o7 c4o5)
)
(and
(g2 c4o6)
(f c4o7 c4o6)
)
(and
(g2 c4o7)
(f c4o7 c4o7)
)
(and
(g2 c4o8)
(f c4o7 c4o8)
)
(and
(g2 c4o9)
(f c4o7 c4o9)
)
(and
(g2 c4o10)
(f c4o7 c4o10)
)
(and
(g2 c4o11)
(f c4o7 c4o11)
)
(and
(g2 c4o12)
(f c4o7 c4o12)
)
)
)
(and
(g1 c4o8)
(or
(and
(g2 c4o1)
(f c4o8 c4o1)
)
(and
(g2 c4o2)
(f c4o8 c4o2)
)
(and
(g2 c4o3)
(f c4o8 c4o3)
)
(and
(g2 c4o4)
(f c4o8 c4o4)
)
(and
(g2 c4o5)
(f c4o8 c4o5)
)
(and
(g2 c4o6)
(f c4o8 c4o6)
)
(and
(g2 c4o7)
(f c4o8 c4o7)
)
(and
(g2 c4o8)
(f c4o8 c4o8)
)
(and
(g2 c4o9)
(f c4o8 c4o9)
)
(and
(g2 c4o10)
(f c4o8 c4o10)
)
(and
(g2 c4o11)
(f c4o8 c4o11)
)
(and
(g2 c4o12)
(f c4o8 c4o12)
)
)
)
(and
(g1 c4o9)
(or
(and
(g2 c4o1)
(f c4o9 c4o1)
)
(and
(g2 c4o2)
(f c4o9 c4o2)
)
(and
(g2 c4o3)
(f c4o9 c4o3)
)
(and
(g2 c4o4)
(f c4o9 c4o4)
)
(and
(g2 c4o5)
(f c4o9 c4o5)
)
(and
(g2 c4o6)
(f c4o9 c4o6)
)
(and
(g2 c4o7)
(f c4o9 c4o7)
)
(and
(g2 c4o8)
(f c4o9 c4o8)
)
(and
(g2 c4o9)
(f c4o9 c4o9)
)
(and
(g2 c4o10)
(f c4o9 c4o10)
)
(and
(g2 c4o11)
(f c4o9 c4o11)
)
(and
(g2 c4o12)
(f c4o9 c4o12)
)
)
)
(and
(g1 c4o10)
(or
(and
(g2 c4o1)
(f c4o10 c4o1)
)
(and
(g2 c4o2)
(f c4o10 c4o2)
)
(and
(g2 c4o3)
(f c4o10 c4o3)
)
(and
(g2 c4o4)
(f c4o10 c4o4)
)
(and
(g2 c4o5)
(f c4o10 c4o5)
)
(and
(g2 c4o6)
(f c4o10 c4o6)
)
(and
(g2 c4o7)
(f c4o10 c4o7)
)
(and
(g2 c4o8)
(f c4o10 c4o8)
)
(and
(g2 c4o9)
(f c4o10 c4o9)
)
(and
(g2 c4o10)
(f c4o10 c4o10)
)
(and
(g2 c4o11)
(f c4o10 c4o11)
)
(and
(g2 c4o12)
(f c4o10 c4o12)
)
)
)
(and
(g1 c4o11)
(or
(and
(g2 c4o1)
(f c4o11 c4o1)
)
(and
(g2 c4o2)
(f c4o11 c4o2)
)
(and
(g2 c4o3)
(f c4o11 c4o3)
)
(and
(g2 c4o4)
(f c4o11 c4o4)
)
(and
(g2 c4o5)
(f c4o11 c4o5)
)
(and
(g2 c4o6)
(f c4o11 c4o6)
)
(and
(g2 c4o7)
(f c4o11 c4o7)
)
(and
(g2 c4o8)
(f c4o11 c4o8)
)
(and
(g2 c4o9)
(f c4o11 c4o9)
)
(and
(g2 c4o10)
(f c4o11 c4o10)
)
(and
(g2 c4o11)
(f c4o11 c4o11)
)
(and
(g2 c4o12)
(f c4o11 c4o12)
)
)
)
(and
(g1 c4o12)
(or
(and
(g2 c4o1)
(f c4o12 c4o1)
)
(and
(g2 c4o2)
(f c4o12 c4o2)
)
(and
(g2 c4o3)
(f c4o12 c4o3)
)
(and
(g2 c4o4)
(f c4o12 c4o4)
)
(and
(g2 c4o5)
(f c4o12 c4o5)
)
(and
(g2 c4o6)
(f c4o12 c4o6)
)
(and
(g2 c4o7)
(f c4o12 c4o7)
)
(and
(g2 c4o8)
(f c4o12 c4o8)
)
(and
(g2 c4o9)
(f c4o12 c4o9)
)
(and
(g2 c4o10)
(f c4o12 c4o10)
)
(and
(g2 c4o11)
(f c4o12 c4o11)
)
(and
(g2 c4o12)
(f c4o12 c4o12)
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
(= c4 false)
(and
(=>
(and
(= q1 true)
(= q2 true)
)
(or
(and
(g1 c4o1)
(or
(and
(g2 c4o1)
(not
(f c4o1 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o1 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o1 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o1 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o1 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o1 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o1 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o1 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o1 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o1 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o1 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o1 c4o12)
)
)
)
)
(and
(g1 c4o2)
(or
(and
(g2 c4o1)
(not
(f c4o2 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o2 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o2 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o2 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o2 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o2 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o2 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o2 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o2 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o2 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o2 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o2 c4o12)
)
)
)
)
(and
(g1 c4o3)
(or
(and
(g2 c4o1)
(not
(f c4o3 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o3 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o3 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o3 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o3 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o3 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o3 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o3 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o3 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o3 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o3 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o3 c4o12)
)
)
)
)
(and
(g1 c4o4)
(or
(and
(g2 c4o1)
(not
(f c4o4 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o4 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o4 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o4 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o4 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o4 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o4 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o4 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o4 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o4 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o4 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o4 c4o12)
)
)
)
)
(and
(g1 c4o5)
(or
(and
(g2 c4o1)
(not
(f c4o5 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o5 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o5 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o5 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o5 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o5 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o5 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o5 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o5 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o5 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o5 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o5 c4o12)
)
)
)
)
(and
(g1 c4o6)
(or
(and
(g2 c4o1)
(not
(f c4o6 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o6 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o6 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o6 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o6 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o6 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o6 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o6 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o6 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o6 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o6 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o6 c4o12)
)
)
)
)
(and
(g1 c4o7)
(or
(and
(g2 c4o1)
(not
(f c4o7 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o7 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o7 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o7 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o7 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o7 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o7 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o7 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o7 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o7 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o7 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o7 c4o12)
)
)
)
)
(and
(g1 c4o8)
(or
(and
(g2 c4o1)
(not
(f c4o8 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o8 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o8 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o8 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o8 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o8 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o8 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o8 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o8 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o8 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o8 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o8 c4o12)
)
)
)
)
(and
(g1 c4o9)
(or
(and
(g2 c4o1)
(not
(f c4o9 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o9 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o9 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o9 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o9 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o9 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o9 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o9 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o9 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o9 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o9 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o9 c4o12)
)
)
)
)
(and
(g1 c4o10)
(or
(and
(g2 c4o1)
(not
(f c4o10 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o10 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o10 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o10 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o10 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o10 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o10 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o10 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o10 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o10 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o10 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o10 c4o12)
)
)
)
)
(and
(g1 c4o11)
(or
(and
(g2 c4o1)
(not
(f c4o11 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o11 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o11 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o11 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o11 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o11 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o11 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o11 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o11 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o11 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o11 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o11 c4o12)
)
)
)
)
(and
(g1 c4o12)
(or
(and
(g2 c4o1)
(not
(f c4o12 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o12 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o12 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o12 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o12 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o12 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o12 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o12 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o12 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o12 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o12 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o12 c4o12)
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
(g1 c4o1)
(and
(=>
(g2 c4o1)
(not
(f c4o1 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o1 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o1 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o1 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o1 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o1 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o1 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o1 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o1 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o1 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o1 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o1 c4o12)
)
)
)
)
(and
(g1 c4o2)
(and
(=>
(g2 c4o1)
(not
(f c4o2 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o2 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o2 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o2 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o2 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o2 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o2 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o2 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o2 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o2 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o2 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o2 c4o12)
)
)
)
)
(and
(g1 c4o3)
(and
(=>
(g2 c4o1)
(not
(f c4o3 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o3 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o3 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o3 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o3 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o3 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o3 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o3 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o3 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o3 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o3 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o3 c4o12)
)
)
)
)
(and
(g1 c4o4)
(and
(=>
(g2 c4o1)
(not
(f c4o4 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o4 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o4 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o4 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o4 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o4 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o4 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o4 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o4 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o4 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o4 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o4 c4o12)
)
)
)
)
(and
(g1 c4o5)
(and
(=>
(g2 c4o1)
(not
(f c4o5 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o5 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o5 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o5 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o5 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o5 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o5 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o5 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o5 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o5 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o5 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o5 c4o12)
)
)
)
)
(and
(g1 c4o6)
(and
(=>
(g2 c4o1)
(not
(f c4o6 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o6 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o6 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o6 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o6 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o6 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o6 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o6 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o6 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o6 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o6 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o6 c4o12)
)
)
)
)
(and
(g1 c4o7)
(and
(=>
(g2 c4o1)
(not
(f c4o7 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o7 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o7 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o7 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o7 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o7 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o7 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o7 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o7 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o7 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o7 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o7 c4o12)
)
)
)
)
(and
(g1 c4o8)
(and
(=>
(g2 c4o1)
(not
(f c4o8 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o8 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o8 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o8 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o8 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o8 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o8 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o8 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o8 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o8 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o8 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o8 c4o12)
)
)
)
)
(and
(g1 c4o9)
(and
(=>
(g2 c4o1)
(not
(f c4o9 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o9 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o9 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o9 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o9 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o9 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o9 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o9 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o9 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o9 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o9 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o9 c4o12)
)
)
)
)
(and
(g1 c4o10)
(and
(=>
(g2 c4o1)
(not
(f c4o10 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o10 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o10 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o10 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o10 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o10 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o10 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o10 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o10 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o10 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o10 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o10 c4o12)
)
)
)
)
(and
(g1 c4o11)
(and
(=>
(g2 c4o1)
(not
(f c4o11 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o11 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o11 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o11 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o11 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o11 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o11 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o11 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o11 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o11 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o11 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o11 c4o12)
)
)
)
)
(and
(g1 c4o12)
(and
(=>
(g2 c4o1)
(not
(f c4o12 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o12 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o12 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o12 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o12 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o12 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o12 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o12 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o12 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o12 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o12 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o12 c4o12)
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
(g1 c4o1)
(or
(and
(g2 c4o1)
(not
(f c4o1 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o1 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o1 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o1 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o1 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o1 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o1 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o1 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o1 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o1 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o1 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o1 c4o12)
)
)
)
)
(=>
(g1 c4o2)
(or
(and
(g2 c4o1)
(not
(f c4o2 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o2 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o2 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o2 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o2 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o2 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o2 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o2 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o2 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o2 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o2 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o2 c4o12)
)
)
)
)
(=>
(g1 c4o3)
(or
(and
(g2 c4o1)
(not
(f c4o3 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o3 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o3 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o3 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o3 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o3 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o3 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o3 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o3 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o3 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o3 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o3 c4o12)
)
)
)
)
(=>
(g1 c4o4)
(or
(and
(g2 c4o1)
(not
(f c4o4 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o4 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o4 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o4 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o4 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o4 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o4 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o4 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o4 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o4 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o4 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o4 c4o12)
)
)
)
)
(=>
(g1 c4o5)
(or
(and
(g2 c4o1)
(not
(f c4o5 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o5 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o5 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o5 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o5 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o5 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o5 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o5 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o5 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o5 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o5 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o5 c4o12)
)
)
)
)
(=>
(g1 c4o6)
(or
(and
(g2 c4o1)
(not
(f c4o6 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o6 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o6 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o6 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o6 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o6 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o6 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o6 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o6 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o6 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o6 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o6 c4o12)
)
)
)
)
(=>
(g1 c4o7)
(or
(and
(g2 c4o1)
(not
(f c4o7 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o7 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o7 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o7 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o7 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o7 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o7 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o7 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o7 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o7 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o7 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o7 c4o12)
)
)
)
)
(=>
(g1 c4o8)
(or
(and
(g2 c4o1)
(not
(f c4o8 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o8 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o8 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o8 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o8 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o8 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o8 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o8 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o8 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o8 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o8 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o8 c4o12)
)
)
)
)
(=>
(g1 c4o9)
(or
(and
(g2 c4o1)
(not
(f c4o9 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o9 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o9 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o9 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o9 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o9 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o9 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o9 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o9 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o9 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o9 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o9 c4o12)
)
)
)
)
(=>
(g1 c4o10)
(or
(and
(g2 c4o1)
(not
(f c4o10 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o10 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o10 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o10 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o10 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o10 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o10 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o10 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o10 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o10 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o10 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o10 c4o12)
)
)
)
)
(=>
(g1 c4o11)
(or
(and
(g2 c4o1)
(not
(f c4o11 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o11 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o11 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o11 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o11 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o11 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o11 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o11 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o11 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o11 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o11 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o11 c4o12)
)
)
)
)
(=>
(g1 c4o12)
(or
(and
(g2 c4o1)
(not
(f c4o12 c4o1)
)
)
(and
(g2 c4o2)
(not
(f c4o12 c4o2)
)
)
(and
(g2 c4o3)
(not
(f c4o12 c4o3)
)
)
(and
(g2 c4o4)
(not
(f c4o12 c4o4)
)
)
(and
(g2 c4o5)
(not
(f c4o12 c4o5)
)
)
(and
(g2 c4o6)
(not
(f c4o12 c4o6)
)
)
(and
(g2 c4o7)
(not
(f c4o12 c4o7)
)
)
(and
(g2 c4o8)
(not
(f c4o12 c4o8)
)
)
(and
(g2 c4o9)
(not
(f c4o12 c4o9)
)
)
(and
(g2 c4o10)
(not
(f c4o12 c4o10)
)
)
(and
(g2 c4o11)
(not
(f c4o12 c4o11)
)
)
(and
(g2 c4o12)
(not
(f c4o12 c4o12)
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
(g1 c4o1)
(and
(=>
(g2 c4o1)
(not
(f c4o1 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o1 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o1 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o1 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o1 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o1 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o1 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o1 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o1 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o1 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o1 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o1 c4o12)
)
)
)
)
(=>
(g1 c4o2)
(and
(=>
(g2 c4o1)
(not
(f c4o2 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o2 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o2 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o2 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o2 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o2 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o2 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o2 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o2 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o2 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o2 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o2 c4o12)
)
)
)
)
(=>
(g1 c4o3)
(and
(=>
(g2 c4o1)
(not
(f c4o3 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o3 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o3 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o3 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o3 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o3 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o3 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o3 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o3 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o3 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o3 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o3 c4o12)
)
)
)
)
(=>
(g1 c4o4)
(and
(=>
(g2 c4o1)
(not
(f c4o4 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o4 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o4 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o4 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o4 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o4 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o4 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o4 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o4 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o4 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o4 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o4 c4o12)
)
)
)
)
(=>
(g1 c4o5)
(and
(=>
(g2 c4o1)
(not
(f c4o5 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o5 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o5 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o5 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o5 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o5 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o5 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o5 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o5 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o5 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o5 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o5 c4o12)
)
)
)
)
(=>
(g1 c4o6)
(and
(=>
(g2 c4o1)
(not
(f c4o6 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o6 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o6 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o6 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o6 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o6 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o6 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o6 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o6 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o6 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o6 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o6 c4o12)
)
)
)
)
(=>
(g1 c4o7)
(and
(=>
(g2 c4o1)
(not
(f c4o7 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o7 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o7 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o7 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o7 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o7 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o7 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o7 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o7 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o7 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o7 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o7 c4o12)
)
)
)
)
(=>
(g1 c4o8)
(and
(=>
(g2 c4o1)
(not
(f c4o8 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o8 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o8 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o8 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o8 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o8 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o8 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o8 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o8 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o8 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o8 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o8 c4o12)
)
)
)
)
(=>
(g1 c4o9)
(and
(=>
(g2 c4o1)
(not
(f c4o9 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o9 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o9 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o9 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o9 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o9 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o9 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o9 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o9 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o9 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o9 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o9 c4o12)
)
)
)
)
(=>
(g1 c4o10)
(and
(=>
(g2 c4o1)
(not
(f c4o10 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o10 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o10 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o10 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o10 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o10 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o10 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o10 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o10 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o10 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o10 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o10 c4o12)
)
)
)
)
(=>
(g1 c4o11)
(and
(=>
(g2 c4o1)
(not
(f c4o11 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o11 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o11 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o11 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o11 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o11 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o11 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o11 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o11 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o11 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o11 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o11 c4o12)
)
)
)
)
(=>
(g1 c4o12)
(and
(=>
(g2 c4o1)
(not
(f c4o12 c4o1)
)
)
(=>
(g2 c4o2)
(not
(f c4o12 c4o2)
)
)
(=>
(g2 c4o3)
(not
(f c4o12 c4o3)
)
)
(=>
(g2 c4o4)
(not
(f c4o12 c4o4)
)
)
(=>
(g2 c4o5)
(not
(f c4o12 c4o5)
)
)
(=>
(g2 c4o6)
(not
(f c4o12 c4o6)
)
)
(=>
(g2 c4o7)
(not
(f c4o12 c4o7)
)
)
(=>
(g2 c4o8)
(not
(f c4o12 c4o8)
)
)
(=>
(g2 c4o9)
(not
(f c4o12 c4o9)
)
)
(=>
(g2 c4o10)
(not
(f c4o12 c4o10)
)
)
(=>
(g2 c4o11)
(not
(f c4o12 c4o11)
)
)
(=>
(g2 c4o12)
(not
(f c4o12 c4o12)
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