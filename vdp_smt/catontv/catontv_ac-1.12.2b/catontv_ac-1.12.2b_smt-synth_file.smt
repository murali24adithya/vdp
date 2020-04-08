(set-logic ALL)

(define-fun t1o1 () (_ BitVec 4) (_ bv0 4))
(define-fun t1o2 () (_ BitVec 4) (_ bv1 4))
(define-fun t1o3 () (_ BitVec 4) (_ bv2 4))
(define-fun t2o1 () (_ BitVec 4) (_ bv3 4))
(define-fun t2o2 () (_ BitVec 4) (_ bv4 4))
(define-fun t2o3 () (_ BitVec 4) (_ bv5 4))
(define-fun c1o1 () (_ BitVec 4) (_ bv6 4))
(define-fun c1o2 () (_ BitVec 4) (_ bv7 4))
(define-fun c1o3 () (_ BitVec 4) (_ bv8 4))
(define-fun c2o1 () (_ BitVec 4) (_ bv9 4))
(define-fun c2o2 () (_ BitVec 4) (_ bv10 4))
(define-fun c2o3 () (_ BitVec 4) (_ bv11 4))
(define-fun c3o1 () (_ BitVec 4) (_ bv12 4))
(define-fun c3o2 () (_ BitVec 4) (_ bv13 4))


(define-fun person () (_ BitVec 2) (_ bv0 2))
(define-fun cat () (_ BitVec 2) (_ bv1 2))
(define-fun remote () (_ BitVec 2) (_ bv2 2))
(define-fun tvmonitor () (_ BitVec 2) (_ bv3 2))


(define-fun labelOf((obj (_ BitVec 4))(lbl (_ BitVec 2))) Bool
(or
(and (= obj t1o1) (= lbl tvmonitor))
(and (= obj t1o2) (= lbl cat))
(and (= obj t1o3) (= lbl cat))
(and (= obj t2o1) (= lbl tvmonitor))
(and (= obj t2o2) (= lbl cat))
(and (= obj t2o3) (= lbl cat))
(and (= obj c1o1) (= lbl tvmonitor))
(and (= obj c1o2) (= lbl cat))
(and (= obj c1o3) (= lbl person))
(and (= obj c2o1) (= lbl remote))
(and (= obj c2o2) (= lbl tvmonitor))
(and (= obj c2o3) (= lbl cat))
(and (= obj c3o1) (= lbl tvmonitor))
(and (= obj c3o2) (= lbl cat))

))

(define-fun leftOf ((x (_ BitVec 4))(y (_ BitVec 4))) Bool
(or
(and (= x c1o1) (= y c1o2))
(and (= x c1o3) (= y c1o2))

))

(define-fun rightOf ((x (_ BitVec 4))(y (_ BitVec 4))) Bool
(or
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o3))

))

(define-fun above ((x (_ BitVec 4))(y (_ BitVec 4))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o3) (= y t1o2))
(and (= x t2o1) (= y t2o2))
(and (= x t2o3) (= y t2o2))

))

(define-fun below ((x (_ BitVec 4))(y (_ BitVec 4))) Bool
(or
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o3))
(and (= x t2o2) (= y t2o1))
(and (= x t2o2) (= y t2o3))

))

(define-fun within ((x (_ BitVec 4))(y (_ BitVec 4))) Bool
(or
(and (= x t1o3) (= y t1o1))
(and (= x t2o3) (= y t2o1))
(and (= x c1o3) (= y c1o1))
(and (= x c2o1) (= y c2o2))
(and (= x c2o1) (= y c2o3))
(and (= x c3o2) (= y c3o1))

))



(define-fun eq ((x1 (_ BitVec 4))(x2 (_ BitVec 4))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 4))(x2 (_ BitVec 4))) Bool (not (= x1 x2)))


(declare-const q1 Bool)

(declare-const lbl_g1 (_ BitVec 2))
(assert (or
(= lbl_g1 person)
(= lbl_g1 cat)
(= lbl_g1 remote)
(= lbl_g1 tvmonitor)
)
)
(define-fun g1 ((x (_ BitVec 4))) Bool
(labelOf x lbl_g1)
)


(declare-const constval_person (_ BitVec 2))
(assert (= constval_person person))
(declare-const constval_cat (_ BitVec 2))
(assert (= constval_cat cat))
(declare-const constval_remote (_ BitVec 2))
(assert (= constval_remote remote))
(declare-const constval_tvmonitor (_ BitVec 2))
(assert (= constval_tvmonitor tvmonitor))

(declare-const eq_x1_x1 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const labelOf_x1_person Bool)
(declare-const labelOf_x1_cat Bool)
(declare-const labelOf_x1_remote Bool)
(declare-const labelOf_x1_tvmonitor Bool)
(declare-const leftOf_x1_x1 Bool)
(declare-const rightOf_x1_x1 Bool)
(declare-const above_x1_x1 Bool)
(declare-const below_x1_x1 Bool)
(declare-const within_x1_x1 Bool)

(define-fun f ((x1 (_ BitVec 4))) Bool
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
labelOf_x1_person
(labelOf x1 person)
)
(=>
labelOf_x1_cat
(labelOf x1 cat)
)
(=>
labelOf_x1_remote
(labelOf x1 remote)
)
(=>
labelOf_x1_tvmonitor
(labelOf x1 tvmonitor)
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
(g1 c1o3)
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
(=>
(g1 c1o3)
(f c1o3)
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
(and
(g1 c1o3)
(f c1o3)
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
(and
(g1 c1o3)
(not
(f c1o3)
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
(=>
(g1 c1o3)
(not
(f c1o3)
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
)
)
)
)
)


(check-sat)
(get-model)