(set-logic ALL)

(define-fun t2o1 () (_ BitVec 5) (_ bv0 5))
(define-fun t2o2 () (_ BitVec 5) (_ bv1 5))
(define-fun t2o3 () (_ BitVec 5) (_ bv2 5))
(define-fun t3o1 () (_ BitVec 5) (_ bv3 5))
(define-fun t3o2 () (_ BitVec 5) (_ bv4 5))
(define-fun t3o3 () (_ BitVec 5) (_ bv5 5))
(define-fun t3o4 () (_ BitVec 5) (_ bv6 5))
(define-fun t3o5 () (_ BitVec 5) (_ bv7 5))
(define-fun t3o6 () (_ BitVec 5) (_ bv8 5))
(define-fun t3o7 () (_ BitVec 5) (_ bv9 5))
(define-fun c1o1 () (_ BitVec 5) (_ bv10 5))
(define-fun c1o2 () (_ BitVec 5) (_ bv11 5))
(define-fun c1o3 () (_ BitVec 5) (_ bv12 5))
(define-fun c1o4 () (_ BitVec 5) (_ bv13 5))
(define-fun c1o5 () (_ BitVec 5) (_ bv14 5))
(define-fun c1o6 () (_ BitVec 5) (_ bv15 5))
(define-fun c2o1 () (_ BitVec 5) (_ bv16 5))
(define-fun c2o2 () (_ BitVec 5) (_ bv17 5))
(define-fun c2o3 () (_ BitVec 5) (_ bv18 5))
(define-fun c2o4 () (_ BitVec 5) (_ bv19 5))
(define-fun c2o5 () (_ BitVec 5) (_ bv20 5))
(define-fun c3o1 () (_ BitVec 5) (_ bv21 5))
(define-fun c3o2 () (_ BitVec 5) (_ bv22 5))
(define-fun c3o3 () (_ BitVec 5) (_ bv23 5))
(define-fun c3o4 () (_ BitVec 5) (_ bv24 5))
(define-fun c3o5 () (_ BitVec 5) (_ bv25 5))


(define-fun dog () (_ BitVec 2) (_ bv0 2))
(define-fun sofa () (_ BitVec 2) (_ bv1 2))
(define-fun bowl () (_ BitVec 2) (_ bv2 2))
(define-fun fork () (_ BitVec 2) (_ bv3 2))
(define-fun bed () (_ BitVec 2) (_ bv4 2))
(define-fun book () (_ BitVec 2) (_ bv5 2))
(define-fun chair () (_ BitVec 2) (_ bv6 2))


(define-fun labelOf((obj (_ BitVec 5))(lbl (_ BitVec 2))) Bool
(or
(and (= obj t2o1) (= lbl sofa))
(and (= obj t2o2) (= lbl dog))
(and (= obj t2o3) (= lbl dog))
(and (= obj t3o1) (= lbl sofa))
(and (= obj t3o2) (= lbl chair))
(and (= obj t3o3) (= lbl sofa))
(and (= obj t3o4) (= lbl chair))
(and (= obj t3o5) (= lbl fork))
(and (= obj t3o6) (= lbl dog))
(and (= obj t3o7) (= lbl dog))
(and (= obj c1o1) (= lbl sofa))
(and (= obj c1o2) (= lbl dog))
(and (= obj c1o3) (= lbl dog))
(and (= obj c1o4) (= lbl dog))
(and (= obj c1o5) (= lbl dog))
(and (= obj c1o6) (= lbl dog))
(and (= obj c2o1) (= lbl book))
(and (= obj c2o2) (= lbl book))
(and (= obj c2o3) (= lbl sofa))
(and (= obj c2o4) (= lbl bowl))
(and (= obj c2o5) (= lbl dog))
(and (= obj c3o1) (= lbl book))
(and (= obj c3o2) (= lbl bed))
(and (= obj c3o3) (= lbl sofa))
(and (= obj c3o4) (= lbl sofa))
(and (= obj c3o5) (= lbl dog))

))

(define-fun leftOf ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t3o1) (= y t3o3))
(and (= x t3o1) (= y t3o4))
(and (= x t3o1) (= y t3o7))
(and (= x t3o2) (= y t3o3))
(and (= x t3o2) (= y t3o4))
(and (= x t3o2) (= y t3o7))
(and (= x t3o5) (= y t3o3))
(and (= x t3o5) (= y t3o4))
(and (= x t3o5) (= y t3o7))
(and (= x t3o6) (= y t3o3))
(and (= x t3o6) (= y t3o4))
(and (= x t3o6) (= y t3o5))
(and (= x t3o6) (= y t3o7))
(and (= x c1o5) (= y c1o2))
(and (= x c1o5) (= y c1o3))
(and (= x c1o5) (= y c1o4))
(and (= x c1o6) (= y c1o3))
(and (= x c1o6) (= y c1o4))
(and (= x c2o1) (= y c2o5))
(and (= x c2o2) (= y c2o1))
(and (= x c2o2) (= y c2o5))
(and (= x c2o4) (= y c2o5))
(and (= x c3o1) (= y c3o2))
(and (= x c3o1) (= y c3o4))
(and (= x c3o5) (= y c3o4))

))

(define-fun rightOf ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t3o3) (= y t3o1))
(and (= x t3o3) (= y t3o2))
(and (= x t3o3) (= y t3o5))
(and (= x t3o3) (= y t3o6))
(and (= x t3o4) (= y t3o1))
(and (= x t3o4) (= y t3o2))
(and (= x t3o4) (= y t3o5))
(and (= x t3o4) (= y t3o6))
(and (= x t3o5) (= y t3o6))
(and (= x t3o7) (= y t3o1))
(and (= x t3o7) (= y t3o2))
(and (= x t3o7) (= y t3o5))
(and (= x t3o7) (= y t3o6))
(and (= x c1o2) (= y c1o5))
(and (= x c1o3) (= y c1o5))
(and (= x c1o3) (= y c1o6))
(and (= x c1o4) (= y c1o5))
(and (= x c1o4) (= y c1o6))
(and (= x c2o1) (= y c2o2))
(and (= x c2o5) (= y c2o1))
(and (= x c2o5) (= y c2o2))
(and (= x c2o5) (= y c2o4))
(and (= x c3o2) (= y c3o1))
(and (= x c3o4) (= y c3o1))
(and (= x c3o4) (= y c3o5))

))

(define-fun above ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t3o6) (= y t3o5))
(and (= x t3o7) (= y t3o5))
(and (= x c1o3) (= y c1o4))
(and (= x c2o2) (= y c2o5))
(and (= x c2o3) (= y c2o5))
(and (= x c2o4) (= y c2o5))
(and (= x c3o5) (= y c3o1))

))

(define-fun below ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t3o5) (= y t3o6))
(and (= x t3o5) (= y t3o7))
(and (= x c1o4) (= y c1o3))
(and (= x c2o5) (= y c2o2))
(and (= x c2o5) (= y c2o3))
(and (= x c2o5) (= y c2o4))
(and (= x c3o1) (= y c3o5))

))

(define-fun within ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t2o2) (= y t2o1))
(and (= x t2o3) (= y t2o1))
(and (= x t3o6) (= y t3o2))
(and (= x t3o7) (= y t3o3))
(and (= x t3o7) (= y t3o4))
(and (= x c1o2) (= y c1o1))
(and (= x c1o3) (= y c1o1))
(and (= x c1o6) (= y c1o1))
(and (= x c2o4) (= y c2o3))
(and (= x c3o5) (= y c3o3))

))



(declare-const q1 Bool)
(assert (= q1 ))
(declare-const q2 Bool)
(assert (= q2 ))

(define-fun g1 ((x (_ BitVec 5))) Bool

)
(define-fun g2 ((x (_ BitVec 5))) Bool

)

(define-fun f ((x1 (_ BitVec 5))(x2 (_ BitVec 5))) Bool

)

(declare-const c1 Bool)
(declare-const c2 Bool)
(declare-const c3 Bool)
(assert (= c true))


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
(g1 t2o1)
(g1 t2o2)
(g1 t2o3)
)
(and
(=>
(g1 t2o1)
(and
(or
(g2 t2o1)
(g2 t2o2)
(g2 t2o3)
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
(g1 t3o5)
(g1 t3o6)
(g1 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o1 t3o5)
)
(=>
(g2 t3o6)
(f t3o1 t3o6)
)
(=>
(g2 t3o7)
(f t3o1 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o2 t3o5)
)
(=>
(g2 t3o6)
(f t3o2 t3o6)
)
(=>
(g2 t3o7)
(f t3o2 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o3 t3o5)
)
(=>
(g2 t3o6)
(f t3o3 t3o6)
)
(=>
(g2 t3o7)
(f t3o3 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o4 t3o5)
)
(=>
(g2 t3o6)
(f t3o4 t3o6)
)
(=>
(g2 t3o7)
(f t3o4 t3o7)
)
)
)
)
(=>
(g1 t3o5)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
)
(and
(=>
(g2 t3o1)
(f t3o5 t3o1)
)
(=>
(g2 t3o2)
(f t3o5 t3o2)
)
(=>
(g2 t3o3)
(f t3o5 t3o3)
)
(=>
(g2 t3o4)
(f t3o5 t3o4)
)
(=>
(g2 t3o5)
(f t3o5 t3o5)
)
(=>
(g2 t3o6)
(f t3o5 t3o6)
)
(=>
(g2 t3o7)
(f t3o5 t3o7)
)
)
)
)
(=>
(g1 t3o6)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
)
(and
(=>
(g2 t3o1)
(f t3o6 t3o1)
)
(=>
(g2 t3o2)
(f t3o6 t3o2)
)
(=>
(g2 t3o3)
(f t3o6 t3o3)
)
(=>
(g2 t3o4)
(f t3o6 t3o4)
)
(=>
(g2 t3o5)
(f t3o6 t3o5)
)
(=>
(g2 t3o6)
(f t3o6 t3o6)
)
(=>
(g2 t3o7)
(f t3o6 t3o7)
)
)
)
)
(=>
(g1 t3o7)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
)
(and
(=>
(g2 t3o1)
(f t3o7 t3o1)
)
(=>
(g2 t3o2)
(f t3o7 t3o2)
)
(=>
(g2 t3o3)
(f t3o7 t3o3)
)
(=>
(g2 t3o4)
(f t3o7 t3o4)
)
(=>
(g2 t3o5)
(f t3o7 t3o5)
)
(=>
(g2 t3o6)
(f t3o7 t3o6)
)
(=>
(g2 t3o7)
(f t3o7 t3o7)
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
(g1 t3o5)
(g1 t3o6)
(g1 t3o7)
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
(and
(g2 t3o5)
(f t3o1 t3o5)
)
(and
(g2 t3o6)
(f t3o1 t3o6)
)
(and
(g2 t3o7)
(f t3o1 t3o7)
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
(and
(g2 t3o5)
(f t3o2 t3o5)
)
(and
(g2 t3o6)
(f t3o2 t3o6)
)
(and
(g2 t3o7)
(f t3o2 t3o7)
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
(and
(g2 t3o5)
(f t3o3 t3o5)
)
(and
(g2 t3o6)
(f t3o3 t3o6)
)
(and
(g2 t3o7)
(f t3o3 t3o7)
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
(and
(g2 t3o5)
(f t3o4 t3o5)
)
(and
(g2 t3o6)
(f t3o4 t3o6)
)
(and
(g2 t3o7)
(f t3o4 t3o7)
)
)
)
(=>
(g1 t3o5)
(or
(and
(g2 t3o1)
(f t3o5 t3o1)
)
(and
(g2 t3o2)
(f t3o5 t3o2)
)
(and
(g2 t3o3)
(f t3o5 t3o3)
)
(and
(g2 t3o4)
(f t3o5 t3o4)
)
(and
(g2 t3o5)
(f t3o5 t3o5)
)
(and
(g2 t3o6)
(f t3o5 t3o6)
)
(and
(g2 t3o7)
(f t3o5 t3o7)
)
)
)
(=>
(g1 t3o6)
(or
(and
(g2 t3o1)
(f t3o6 t3o1)
)
(and
(g2 t3o2)
(f t3o6 t3o2)
)
(and
(g2 t3o3)
(f t3o6 t3o3)
)
(and
(g2 t3o4)
(f t3o6 t3o4)
)
(and
(g2 t3o5)
(f t3o6 t3o5)
)
(and
(g2 t3o6)
(f t3o6 t3o6)
)
(and
(g2 t3o7)
(f t3o6 t3o7)
)
)
)
(=>
(g1 t3o7)
(or
(and
(g2 t3o1)
(f t3o7 t3o1)
)
(and
(g2 t3o2)
(f t3o7 t3o2)
)
(and
(g2 t3o3)
(f t3o7 t3o3)
)
(and
(g2 t3o4)
(f t3o7 t3o4)
)
(and
(g2 t3o5)
(f t3o7 t3o5)
)
(and
(g2 t3o6)
(f t3o7 t3o6)
)
(and
(g2 t3o7)
(f t3o7 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o1 t3o5)
)
(=>
(g2 t3o6)
(f t3o1 t3o6)
)
(=>
(g2 t3o7)
(f t3o1 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o2 t3o5)
)
(=>
(g2 t3o6)
(f t3o2 t3o6)
)
(=>
(g2 t3o7)
(f t3o2 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o3 t3o5)
)
(=>
(g2 t3o6)
(f t3o3 t3o6)
)
(=>
(g2 t3o7)
(f t3o3 t3o7)
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
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
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
(=>
(g2 t3o5)
(f t3o4 t3o5)
)
(=>
(g2 t3o6)
(f t3o4 t3o6)
)
(=>
(g2 t3o7)
(f t3o4 t3o7)
)
)
)
)
(and
(g1 t3o5)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
)
(and
(=>
(g2 t3o1)
(f t3o5 t3o1)
)
(=>
(g2 t3o2)
(f t3o5 t3o2)
)
(=>
(g2 t3o3)
(f t3o5 t3o3)
)
(=>
(g2 t3o4)
(f t3o5 t3o4)
)
(=>
(g2 t3o5)
(f t3o5 t3o5)
)
(=>
(g2 t3o6)
(f t3o5 t3o6)
)
(=>
(g2 t3o7)
(f t3o5 t3o7)
)
)
)
)
(and
(g1 t3o6)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
)
(and
(=>
(g2 t3o1)
(f t3o6 t3o1)
)
(=>
(g2 t3o2)
(f t3o6 t3o2)
)
(=>
(g2 t3o3)
(f t3o6 t3o3)
)
(=>
(g2 t3o4)
(f t3o6 t3o4)
)
(=>
(g2 t3o5)
(f t3o6 t3o5)
)
(=>
(g2 t3o6)
(f t3o6 t3o6)
)
(=>
(g2 t3o7)
(f t3o6 t3o7)
)
)
)
)
(and
(g1 t3o7)
(and
(or
(g2 t3o1)
(g2 t3o2)
(g2 t3o3)
(g2 t3o4)
(g2 t3o5)
(g2 t3o6)
(g2 t3o7)
)
(and
(=>
(g2 t3o1)
(f t3o7 t3o1)
)
(=>
(g2 t3o2)
(f t3o7 t3o2)
)
(=>
(g2 t3o3)
(f t3o7 t3o3)
)
(=>
(g2 t3o4)
(f t3o7 t3o4)
)
(=>
(g2 t3o5)
(f t3o7 t3o5)
)
(=>
(g2 t3o6)
(f t3o7 t3o6)
)
(=>
(g2 t3o7)
(f t3o7 t3o7)
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
(and
(g2 t3o5)
(f t3o1 t3o5)
)
(and
(g2 t3o6)
(f t3o1 t3o6)
)
(and
(g2 t3o7)
(f t3o1 t3o7)
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
(and
(g2 t3o5)
(f t3o2 t3o5)
)
(and
(g2 t3o6)
(f t3o2 t3o6)
)
(and
(g2 t3o7)
(f t3o2 t3o7)
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
(and
(g2 t3o5)
(f t3o3 t3o5)
)
(and
(g2 t3o6)
(f t3o3 t3o6)
)
(and
(g2 t3o7)
(f t3o3 t3o7)
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
(and
(g2 t3o5)
(f t3o4 t3o5)
)
(and
(g2 t3o6)
(f t3o4 t3o6)
)
(and
(g2 t3o7)
(f t3o4 t3o7)
)
)
)
(and
(g1 t3o5)
(or
(and
(g2 t3o1)
(f t3o5 t3o1)
)
(and
(g2 t3o2)
(f t3o5 t3o2)
)
(and
(g2 t3o3)
(f t3o5 t3o3)
)
(and
(g2 t3o4)
(f t3o5 t3o4)
)
(and
(g2 t3o5)
(f t3o5 t3o5)
)
(and
(g2 t3o6)
(f t3o5 t3o6)
)
(and
(g2 t3o7)
(f t3o5 t3o7)
)
)
)
(and
(g1 t3o6)
(or
(and
(g2 t3o1)
(f t3o6 t3o1)
)
(and
(g2 t3o2)
(f t3o6 t3o2)
)
(and
(g2 t3o3)
(f t3o6 t3o3)
)
(and
(g2 t3o4)
(f t3o6 t3o4)
)
(and
(g2 t3o5)
(f t3o6 t3o5)
)
(and
(g2 t3o6)
(f t3o6 t3o6)
)
(and
(g2 t3o7)
(f t3o6 t3o7)
)
)
)
(and
(g1 t3o7)
(or
(and
(g2 t3o1)
(f t3o7 t3o1)
)
(and
(g2 t3o2)
(f t3o7 t3o2)
)
(and
(g2 t3o3)
(f t3o7 t3o3)
)
(and
(g2 t3o4)
(f t3o7 t3o4)
)
(and
(g2 t3o5)
(f t3o7 t3o5)
)
(and
(g2 t3o6)
(f t3o7 t3o6)
)
(and
(g2 t3o7)
(f t3o7 t3o7)
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
(g1 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o1 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o2 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o3 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o4 c2o5)
)
)
)
)
(=>
(g1 c2o5)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
(g2 c2o5)
)
(and
(=>
(g2 c2o1)
(f c2o5 c2o1)
)
(=>
(g2 c2o2)
(f c2o5 c2o2)
)
(=>
(g2 c2o3)
(f c2o5 c2o3)
)
(=>
(g2 c2o4)
(f c2o5 c2o4)
)
(=>
(g2 c2o5)
(f c2o5 c2o5)
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
(g1 c2o5)
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
(and
(g2 c2o5)
(f c2o1 c2o5)
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
(and
(g2 c2o5)
(f c2o2 c2o5)
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
(and
(g2 c2o5)
(f c2o3 c2o5)
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
(and
(g2 c2o5)
(f c2o4 c2o5)
)
)
)
(=>
(g1 c2o5)
(or
(and
(g2 c2o1)
(f c2o5 c2o1)
)
(and
(g2 c2o2)
(f c2o5 c2o2)
)
(and
(g2 c2o3)
(f c2o5 c2o3)
)
(and
(g2 c2o4)
(f c2o5 c2o4)
)
(and
(g2 c2o5)
(f c2o5 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o1 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o2 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o3 c2o5)
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
(g2 c2o5)
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
(=>
(g2 c2o5)
(f c2o4 c2o5)
)
)
)
)
(and
(g1 c2o5)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
(g2 c2o4)
(g2 c2o5)
)
(and
(=>
(g2 c2o1)
(f c2o5 c2o1)
)
(=>
(g2 c2o2)
(f c2o5 c2o2)
)
(=>
(g2 c2o3)
(f c2o5 c2o3)
)
(=>
(g2 c2o4)
(f c2o5 c2o4)
)
(=>
(g2 c2o5)
(f c2o5 c2o5)
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
(and
(g2 c2o5)
(f c2o1 c2o5)
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
(and
(g2 c2o5)
(f c2o2 c2o5)
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
(and
(g2 c2o5)
(f c2o3 c2o5)
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
(and
(g2 c2o5)
(f c2o4 c2o5)
)
)
)
(and
(g1 c2o5)
(or
(and
(g2 c2o1)
(f c2o5 c2o1)
)
(and
(g2 c2o2)
(f c2o5 c2o2)
)
(and
(g2 c2o3)
(f c2o5 c2o3)
)
(and
(g2 c2o4)
(f c2o5 c2o4)
)
(and
(g2 c2o5)
(f c2o5 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o1 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o2 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o3 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o4 c2o5)
)
)
)
)
(and
(g1 c2o5)
(or
(and
(g2 c2o1)
(not
(f c2o5 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o5 c2o2)
)
)
(and
(g2 c2o3)
(not
(f c2o5 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o5 c2o4)
)
)
(and
(g2 c2o5)
(not
(f c2o5 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o1 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o2 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o3 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o4 c2o5)
)
)
)
)
(and
(g1 c2o5)
(and
(=>
(g2 c2o1)
(not
(f c2o5 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o5 c2o2)
)
)
(=>
(g2 c2o3)
(not
(f c2o5 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o5 c2o4)
)
)
(=>
(g2 c2o5)
(not
(f c2o5 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o1 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o2 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o3 c2o5)
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
(and
(g2 c2o5)
(not
(f c2o4 c2o5)
)
)
)
)
(=>
(g1 c2o5)
(or
(and
(g2 c2o1)
(not
(f c2o5 c2o1)
)
)
(and
(g2 c2o2)
(not
(f c2o5 c2o2)
)
)
(and
(g2 c2o3)
(not
(f c2o5 c2o3)
)
)
(and
(g2 c2o4)
(not
(f c2o5 c2o4)
)
)
(and
(g2 c2o5)
(not
(f c2o5 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o1 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o2 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o3 c2o5)
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
(=>
(g2 c2o5)
(not
(f c2o4 c2o5)
)
)
)
)
(=>
(g1 c2o5)
(and
(=>
(g2 c2o1)
(not
(f c2o5 c2o1)
)
)
(=>
(g2 c2o2)
(not
(f c2o5 c2o2)
)
)
(=>
(g2 c2o3)
(not
(f c2o5 c2o3)
)
)
(=>
(g2 c2o4)
(not
(f c2o5 c2o4)
)
)
(=>
(g2 c2o5)
(not
(f c2o5 c2o5)
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
(g1 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o1 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o2 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o3 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o4 c3o5)
)
)
)
)
(=>
(g1 c3o5)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
)
(and
(=>
(g2 c3o1)
(f c3o5 c3o1)
)
(=>
(g2 c3o2)
(f c3o5 c3o2)
)
(=>
(g2 c3o3)
(f c3o5 c3o3)
)
(=>
(g2 c3o4)
(f c3o5 c3o4)
)
(=>
(g2 c3o5)
(f c3o5 c3o5)
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
(g1 c3o5)
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
(and
(g2 c3o5)
(f c3o1 c3o5)
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
(and
(g2 c3o5)
(f c3o2 c3o5)
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
(and
(g2 c3o5)
(f c3o3 c3o5)
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
(and
(g2 c3o5)
(f c3o4 c3o5)
)
)
)
(=>
(g1 c3o5)
(or
(and
(g2 c3o1)
(f c3o5 c3o1)
)
(and
(g2 c3o2)
(f c3o5 c3o2)
)
(and
(g2 c3o3)
(f c3o5 c3o3)
)
(and
(g2 c3o4)
(f c3o5 c3o4)
)
(and
(g2 c3o5)
(f c3o5 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o1 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o2 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o3 c3o5)
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
(g2 c3o5)
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
(=>
(g2 c3o5)
(f c3o4 c3o5)
)
)
)
)
(and
(g1 c3o5)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
)
(and
(=>
(g2 c3o1)
(f c3o5 c3o1)
)
(=>
(g2 c3o2)
(f c3o5 c3o2)
)
(=>
(g2 c3o3)
(f c3o5 c3o3)
)
(=>
(g2 c3o4)
(f c3o5 c3o4)
)
(=>
(g2 c3o5)
(f c3o5 c3o5)
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
(and
(g2 c3o5)
(f c3o1 c3o5)
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
(and
(g2 c3o5)
(f c3o2 c3o5)
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
(and
(g2 c3o5)
(f c3o3 c3o5)
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
(and
(g2 c3o5)
(f c3o4 c3o5)
)
)
)
(and
(g1 c3o5)
(or
(and
(g2 c3o1)
(f c3o5 c3o1)
)
(and
(g2 c3o2)
(f c3o5 c3o2)
)
(and
(g2 c3o3)
(f c3o5 c3o3)
)
(and
(g2 c3o4)
(f c3o5 c3o4)
)
(and
(g2 c3o5)
(f c3o5 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o1 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o2 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o3 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o4 c3o5)
)
)
)
)
(and
(g1 c3o5)
(or
(and
(g2 c3o1)
(not
(f c3o5 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o5 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o5 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o5 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o5 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o1 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o2 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o3 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o4 c3o5)
)
)
)
)
(and
(g1 c3o5)
(and
(=>
(g2 c3o1)
(not
(f c3o5 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o5 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o5 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o5 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o5 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o1 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o2 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o3 c3o5)
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
(and
(g2 c3o5)
(not
(f c3o4 c3o5)
)
)
)
)
(=>
(g1 c3o5)
(or
(and
(g2 c3o1)
(not
(f c3o5 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o5 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o5 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o5 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o5 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o1 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o2 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o3 c3o5)
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
(=>
(g2 c3o5)
(not
(f c3o4 c3o5)
)
)
)
)
(=>
(g1 c3o5)
(and
(=>
(g2 c3o1)
(not
(f c3o5 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o5 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o5 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o5 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o5 c3o5)
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