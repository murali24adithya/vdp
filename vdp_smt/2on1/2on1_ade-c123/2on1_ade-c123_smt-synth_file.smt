(set-logic ALL)

(define-fun t1o1 () (_ BitVec 5) (_ bv0 5))
(define-fun t1o2 () (_ BitVec 5) (_ bv1 5))
(define-fun t1o3 () (_ BitVec 5) (_ bv2 5))
(define-fun t2o1 () (_ BitVec 5) (_ bv3 5))
(define-fun t2o2 () (_ BitVec 5) (_ bv4 5))
(define-fun t2o3 () (_ BitVec 5) (_ bv5 5))
(define-fun t3o1 () (_ BitVec 5) (_ bv6 5))
(define-fun t3o2 () (_ BitVec 5) (_ bv7 5))
(define-fun t3o3 () (_ BitVec 5) (_ bv8 5))
(define-fun t3o4 () (_ BitVec 5) (_ bv9 5))
(define-fun c1o1 () (_ BitVec 5) (_ bv10 5))
(define-fun c1o2 () (_ BitVec 5) (_ bv11 5))
(define-fun c1o3 () (_ BitVec 5) (_ bv12 5))
(define-fun c1o4 () (_ BitVec 5) (_ bv13 5))
(define-fun c2o1 () (_ BitVec 5) (_ bv14 5))
(define-fun c2o2 () (_ BitVec 5) (_ bv15 5))
(define-fun c2o3 () (_ BitVec 5) (_ bv16 5))
(define-fun c3o1 () (_ BitVec 5) (_ bv17 5))
(define-fun c3o2 () (_ BitVec 5) (_ bv18 5))
(define-fun c4o1 () (_ BitVec 5) (_ bv19 5))
(define-fun c4o2 () (_ BitVec 5) (_ bv20 5))
(define-fun c4o3 () (_ BitVec 5) (_ bv21 5))


(define-fun cat () (_ BitVec 2) (_ bv0 2))
(define-fun chair () (_ BitVec 2) (_ bv1 2))
(define-fun sofa () (_ BitVec 2) (_ bv2 2))
(define-fun bed () (_ BitVec 2) (_ bv3 2))


(define-fun labelOf((obj (_ BitVec 5))(lbl (_ BitVec 2))) Bool
(or
(and (= obj t1o1) (= lbl sofa))
(and (= obj t1o2) (= lbl cat))
(and (= obj t1o3) (= lbl cat))
(and (= obj t2o1) (= lbl sofa))
(and (= obj t2o2) (= lbl cat))
(and (= obj t2o3) (= lbl cat))
(and (= obj t3o1) (= lbl bed))
(and (= obj t3o2) (= lbl sofa))
(and (= obj t3o3) (= lbl cat))
(and (= obj t3o4) (= lbl cat))
(and (= obj c1o1) (= lbl chair))
(and (= obj c1o1) (= lbl sofa))
(and (= obj c1o2) (= lbl chair))
(and (= obj c1o3) (= lbl cat))
(and (= obj c1o4) (= lbl cat))
(and (= obj c2o1) (= lbl sofa))
(and (= obj c2o2) (= lbl cat))
(and (= obj c2o3) (= lbl cat))
(and (= obj c3o1) (= lbl sofa))
(and (= obj c3o2) (= lbl cat))
(and (= obj c4o1) (= lbl sofa))
(and (= obj c4o2) (= lbl cat))
(and (= obj c4o3) (= lbl cat))

))

(define-fun leftOf ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o3) (= y t1o2))
(and (= x t2o2) (= y t2o3))
(and (= x t3o4) (= y t3o3))
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o4))
(and (= x c1o3) (= y c1o1))
(and (= x c1o3) (= y c1o4))
(and (= x c2o2) (= y c2o1))
(and (= x c2o2) (= y c2o3))
(and (= x c4o2) (= y c4o3))

))

(define-fun rightOf ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o2) (= y t1o3))
(and (= x t2o3) (= y t2o2))
(and (= x t3o3) (= y t3o4))
(and (= x c1o1) (= y c1o2))
(and (= x c1o1) (= y c1o3))
(and (= x c1o4) (= y c1o2))
(and (= x c1o4) (= y c1o3))
(and (= x c2o1) (= y c2o2))
(and (= x c2o3) (= y c2o2))
(and (= x c4o3) (= y c4o2))

))

(define-fun above ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
false
))

(define-fun below ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
false
))

(define-fun within ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o2) (= y t1o1))
(and (= x t1o3) (= y t1o1))
(and (= x t2o2) (= y t2o1))
(and (= x t2o3) (= y t2o1))
(and (= x t3o1) (= y t3o2))
(and (= x t3o3) (= y t3o1))
(and (= x t3o3) (= y t3o2))
(and (= x t3o4) (= y t3o1))
(and (= x t3o4) (= y t3o2))
(and (= x c1o3) (= y c1o2))
(and (= x c1o4) (= y c1o1))
(and (= x c2o3) (= y c2o1))
(and (= x c3o2) (= y c3o1))
(and (= x c4o2) (= y c4o1))
(and (= x c4o3) (= y c4o1))

))



(define-fun eq ((x1 (_ BitVec 5))(x2 (_ BitVec 5))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 5))(x2 (_ BitVec 5))) Bool (not (= x1 x2)))


(declare-const q1 Bool)
(declare-const q2 Bool)
(declare-const q3 Bool)

(declare-const lbl_g1 (_ BitVec 2))
(assert (or
(= lbl_g1 cat)
(= lbl_g1 chair)
(= lbl_g1 sofa)
(= lbl_g1 bed)
)
)
(define-fun g1 ((x (_ BitVec 5))) Bool
(labelOf x lbl_g1)
)

(declare-const lbl_g2 (_ BitVec 2))
(assert (or
(= lbl_g2 cat)
(= lbl_g2 chair)
(= lbl_g2 sofa)
(= lbl_g2 bed)
)
)
(define-fun g2 ((x (_ BitVec 5))) Bool
(labelOf x lbl_g2)
)

(declare-const lbl_g3 (_ BitVec 2))
(assert (or
(= lbl_g3 cat)
(= lbl_g3 chair)
(= lbl_g3 sofa)
(= lbl_g3 bed)
)
)
(define-fun g3 ((x (_ BitVec 5))) Bool
(labelOf x lbl_g3)
)


(declare-const constval_cat (_ BitVec 2))
(assert (= constval_cat cat))
(declare-const constval_chair (_ BitVec 2))
(assert (= constval_chair chair))
(declare-const constval_sofa (_ BitVec 2))
(assert (= constval_sofa sofa))
(declare-const constval_bed (_ BitVec 2))
(assert (= constval_bed bed))

(declare-const eq_x1_x1 Bool)
(declare-const eq_x1_x2 Bool)
(declare-const eq_x1_x3 Bool)
(declare-const eq_x2_x1 Bool)
(declare-const eq_x2_x2 Bool)
(declare-const eq_x2_x3 Bool)
(declare-const eq_x3_x1 Bool)
(declare-const eq_x3_x2 Bool)
(declare-const eq_x3_x3 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const neq_x1_x2 Bool)
(declare-const neq_x1_x3 Bool)
(declare-const neq_x2_x1 Bool)
(declare-const neq_x2_x2 Bool)
(declare-const neq_x2_x3 Bool)
(declare-const neq_x3_x1 Bool)
(declare-const neq_x3_x2 Bool)
(declare-const neq_x3_x3 Bool)
(declare-const labelOf_x1_cat Bool)
(declare-const labelOf_x1_chair Bool)
(declare-const labelOf_x1_sofa Bool)
(declare-const labelOf_x1_bed Bool)
(declare-const labelOf_x2_cat Bool)
(declare-const labelOf_x2_chair Bool)
(declare-const labelOf_x2_sofa Bool)
(declare-const labelOf_x2_bed Bool)
(declare-const labelOf_x3_cat Bool)
(declare-const labelOf_x3_chair Bool)
(declare-const labelOf_x3_sofa Bool)
(declare-const labelOf_x3_bed Bool)
(declare-const leftOf_x1_x1 Bool)
(declare-const leftOf_x1_x2 Bool)
(declare-const leftOf_x1_x3 Bool)
(declare-const leftOf_x2_x1 Bool)
(declare-const leftOf_x2_x2 Bool)
(declare-const leftOf_x2_x3 Bool)
(declare-const leftOf_x3_x1 Bool)
(declare-const leftOf_x3_x2 Bool)
(declare-const leftOf_x3_x3 Bool)
(declare-const rightOf_x1_x1 Bool)
(declare-const rightOf_x1_x2 Bool)
(declare-const rightOf_x1_x3 Bool)
(declare-const rightOf_x2_x1 Bool)
(declare-const rightOf_x2_x2 Bool)
(declare-const rightOf_x2_x3 Bool)
(declare-const rightOf_x3_x1 Bool)
(declare-const rightOf_x3_x2 Bool)
(declare-const rightOf_x3_x3 Bool)
(declare-const above_x1_x1 Bool)
(declare-const above_x1_x2 Bool)
(declare-const above_x1_x3 Bool)
(declare-const above_x2_x1 Bool)
(declare-const above_x2_x2 Bool)
(declare-const above_x2_x3 Bool)
(declare-const above_x3_x1 Bool)
(declare-const above_x3_x2 Bool)
(declare-const above_x3_x3 Bool)
(declare-const below_x1_x1 Bool)
(declare-const below_x1_x2 Bool)
(declare-const below_x1_x3 Bool)
(declare-const below_x2_x1 Bool)
(declare-const below_x2_x2 Bool)
(declare-const below_x2_x3 Bool)
(declare-const below_x3_x1 Bool)
(declare-const below_x3_x2 Bool)
(declare-const below_x3_x3 Bool)
(declare-const within_x1_x1 Bool)
(declare-const within_x1_x2 Bool)
(declare-const within_x1_x3 Bool)
(declare-const within_x2_x1 Bool)
(declare-const within_x2_x2 Bool)
(declare-const within_x2_x3 Bool)
(declare-const within_x3_x1 Bool)
(declare-const within_x3_x2 Bool)
(declare-const within_x3_x3 Bool)

(define-fun f ((x1 (_ BitVec 5)) (x2 (_ BitVec 5)) (x3 (_ BitVec 5))) Bool
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
eq_x1_x3
(eq x1 x3)
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
eq_x2_x3
(eq x2 x3)
)
(=>
eq_x3_x1
(eq x3 x1)
)
(=>
eq_x3_x2
(eq x3 x2)
)
(=>
eq_x3_x3
(eq x3 x3)
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
neq_x1_x3
(neq x1 x3)
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
neq_x2_x3
(neq x2 x3)
)
(=>
neq_x3_x1
(neq x3 x1)
)
(=>
neq_x3_x2
(neq x3 x2)
)
(=>
neq_x3_x3
(neq x3 x3)
)
(=>
labelOf_x1_cat
(labelOf x1 cat)
)
(=>
labelOf_x1_chair
(labelOf x1 chair)
)
(=>
labelOf_x1_sofa
(labelOf x1 sofa)
)
(=>
labelOf_x1_bed
(labelOf x1 bed)
)
(=>
labelOf_x2_cat
(labelOf x2 cat)
)
(=>
labelOf_x2_chair
(labelOf x2 chair)
)
(=>
labelOf_x2_sofa
(labelOf x2 sofa)
)
(=>
labelOf_x2_bed
(labelOf x2 bed)
)
(=>
labelOf_x3_cat
(labelOf x3 cat)
)
(=>
labelOf_x3_chair
(labelOf x3 chair)
)
(=>
labelOf_x3_sofa
(labelOf x3 sofa)
)
(=>
labelOf_x3_bed
(labelOf x3 bed)
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
leftOf_x1_x3
(leftOf x1 x3)
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
leftOf_x2_x3
(leftOf x2 x3)
)
(=>
leftOf_x3_x1
(leftOf x3 x1)
)
(=>
leftOf_x3_x2
(leftOf x3 x2)
)
(=>
leftOf_x3_x3
(leftOf x3 x3)
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
rightOf_x1_x3
(rightOf x1 x3)
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
rightOf_x2_x3
(rightOf x2 x3)
)
(=>
rightOf_x3_x1
(rightOf x3 x1)
)
(=>
rightOf_x3_x2
(rightOf x3 x2)
)
(=>
rightOf_x3_x3
(rightOf x3 x3)
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
above_x1_x3
(above x1 x3)
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
above_x2_x3
(above x2 x3)
)
(=>
above_x3_x1
(above x3 x1)
)
(=>
above_x3_x2
(above x3 x2)
)
(=>
above_x3_x3
(above x3 x3)
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
below_x1_x3
(below x1 x3)
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
below_x2_x3
(below x2 x3)
)
(=>
below_x3_x1
(below x3 x1)
)
(=>
below_x3_x2
(below x3 x2)
)
(=>
below_x3_x3
(below x3 x3)
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
within_x1_x3
(within x1 x3)
)
(=>
within_x2_x1
(within x2 x1)
)
(=>
within_x2_x2
(within x2 x2)
)
(=>
within_x2_x3
(within x2 x3)
)
(=>
within_x3_x1
(within x3 x1)
)
(=>
within_x3_x2
(within x3 x2)
)
(=>
within_x3_x3
(within x3 x3)
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
(= q3 true)
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
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
)
(and
(=>
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
)
(=>
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
)
(=>
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
)
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
)
(and
(=>
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
)
(=>
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
)
(=>
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
)
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
)
(and
(=>
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
)
(=>
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
)
(=>
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
)
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
(= q2 true)
(= q3 false)
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
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
)
(and
(=>
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
(=>
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
(=>
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
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
)
(and
(=>
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
(=>
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
(=>
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
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
)
(and
(=>
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
(=>
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
(=>
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
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
(= q3 true)
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
(or
(and
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
)
(and
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
)
(and
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
)
)
)
)
(=>
(g1 t1o2)
(or
(and
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
)
(and
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
)
(and
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
)
)
)
)
(=>
(g1 t1o3)
(or
(and
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
)
(and
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
)
(and
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
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
(= q3 false)
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
(or
(and
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
(and
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
(and
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
)
)
)
(=>
(g1 t1o2)
(or
(and
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
(and
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
(and
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
)
)
)
(=>
(g1 t1o3)
(or
(and
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
(and
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
(and
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 t1o1)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
)
(and
(=>
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
)
(=>
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
)
(=>
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
)
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
)
(and
(=>
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
)
(=>
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
)
(=>
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
)
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
)
(and
(=>
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
)
(=>
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
)
(=>
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
)
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
(= q3 false)
)
(or
(and
(g1 t1o1)
(and
(or
(g2 t1o1)
(g2 t1o2)
(g2 t1o3)
)
(and
(=>
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
(=>
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
(=>
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
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
)
(and
(=>
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
(=>
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
(=>
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
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
)
(and
(=>
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
(=>
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
(=>
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 t1o1)
(or
(and
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
)
(and
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
)
(and
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
)
)
)
)
(and
(g1 t1o2)
(or
(and
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
)
(and
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
)
(and
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
)
)
)
)
(and
(g1 t1o3)
(or
(and
(g2 t1o1)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
)
(and
(g2 t1o2)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
)
(and
(g2 t1o3)
(and
(or
(g3 t1o1)
(g3 t1o2)
(g3 t1o3)
)
(and
(=>
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(=>
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(=>
(g3 t1o3)
(f t1o3 t1o3 t1o3)
)
)
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
(= q3 false)
)
(or
(and
(g1 t1o1)
(or
(and
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o1 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o1 t1o3)
)
)
)
(and
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o1 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o2 t1o3)
)
)
)
(and
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o1 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o1 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o1 t1o3 t1o3)
)
)
)
)
)
(and
(g1 t1o2)
(or
(and
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o2 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o1 t1o3)
)
)
)
(and
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o2 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o2 t1o3)
)
)
)
(and
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o2 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o2 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o2 t1o3 t1o3)
)
)
)
)
)
(and
(g1 t1o3)
(or
(and
(g2 t1o1)
(or
(and
(g3 t1o1)
(f t1o3 t1o1 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o1 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o1 t1o3)
)
)
)
(and
(g2 t1o2)
(or
(and
(g3 t1o1)
(f t1o3 t1o2 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o2 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o2 t1o3)
)
)
)
(and
(g2 t1o3)
(or
(and
(g3 t1o1)
(f t1o3 t1o3 t1o1)
)
(and
(g3 t1o2)
(f t1o3 t1o3 t1o2)
)
(and
(g3 t1o3)
(f t1o3 t1o3 t1o3)
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
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
)
(=>
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
)
(=>
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
)
(=>
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
)
(=>
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
)
(=>
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
)
(=>
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
)
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
(= q2 true)
(= q3 false)
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
(or
(and
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
(=>
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
(=>
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
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
(or
(and
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
(=>
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
(=>
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
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
(or
(and
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
(=>
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
(=>
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
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
(= q3 true)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
)
(and
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
)
(and
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
)
)
)
)
(=>
(g1 t2o2)
(or
(and
(g2 t2o1)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
)
(and
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
)
(and
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
)
)
)
)
(=>
(g1 t2o3)
(or
(and
(g2 t2o1)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
)
(and
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
)
(and
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
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
(= q3 false)
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
(or
(and
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
(and
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
(and
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
)
)
)
(=>
(g1 t2o2)
(or
(and
(g2 t2o1)
(or
(and
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
(and
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
(and
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
)
)
)
(=>
(g1 t2o3)
(or
(and
(g2 t2o1)
(or
(and
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
(and
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
(and
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
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
(= q3 true)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
)
(=>
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
)
(=>
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
)
(=>
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
)
(=>
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
)
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
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
)
(=>
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
)
(=>
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
)
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
(= q3 false)
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
(or
(and
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
(=>
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
(=>
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
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
(or
(and
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
(=>
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
(=>
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
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
(or
(and
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
(=>
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
(=>
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 t2o1)
(or
(and
(g2 t2o1)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
)
(and
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
)
(and
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
)
)
)
)
(and
(g1 t2o2)
(or
(and
(g2 t2o1)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
)
(and
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
)
(and
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
)
)
)
)
(and
(g1 t2o3)
(or
(and
(g2 t2o1)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
)
(and
(g2 t2o2)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
)
(and
(g2 t2o3)
(and
(or
(g3 t2o1)
(g3 t2o2)
(g3 t2o3)
)
(and
(=>
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(=>
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(=>
(g3 t2o3)
(f t2o3 t2o3 t2o3)
)
)
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
(= q3 false)
)
(or
(and
(g1 t2o1)
(or
(and
(g2 t2o1)
(or
(and
(g3 t2o1)
(f t2o1 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o1 t2o3)
)
)
)
(and
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o1 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o2 t2o3)
)
)
)
(and
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o1 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o1 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o1 t2o3 t2o3)
)
)
)
)
)
(and
(g1 t2o2)
(or
(and
(g2 t2o1)
(or
(and
(g3 t2o1)
(f t2o2 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o1 t2o3)
)
)
)
(and
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o2 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o2 t2o3)
)
)
)
(and
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o2 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o2 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o2 t2o3 t2o3)
)
)
)
)
)
(and
(g1 t2o3)
(or
(and
(g2 t2o1)
(or
(and
(g3 t2o1)
(f t2o3 t2o1 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o1 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o1 t2o3)
)
)
)
(and
(g2 t2o2)
(or
(and
(g3 t2o1)
(f t2o3 t2o2 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o2 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o2 t2o3)
)
)
)
(and
(g2 t2o3)
(or
(and
(g3 t2o1)
(f t2o3 t2o3 t2o1)
)
(and
(g3 t2o2)
(f t2o3 t2o3 t2o2)
)
(and
(g3 t2o3)
(f t2o3 t2o3 t2o3)
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
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
)
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
(= q2 true)
(= q3 false)
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
(or
(and
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
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
(or
(and
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
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
(or
(and
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
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
(or
(and
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
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
(= q3 true)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
)
)
)
)
(=>
(g1 t3o2)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
)
)
)
)
(=>
(g1 t3o3)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
)
)
)
)
(=>
(g1 t3o4)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
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
(= q3 false)
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
(or
(and
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
)
)
)
(=>
(g1 t3o2)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
)
)
)
(=>
(g1 t3o3)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
)
)
)
(=>
(g1 t3o4)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
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
(= q3 true)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
)
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
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
)
(=>
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
)
(=>
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
)
(=>
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
)
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
(= q3 false)
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
(or
(and
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
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
(or
(and
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
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
(or
(and
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
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
(or
(and
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
(=>
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
(=>
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
(=>
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
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
(= q3 true)
)
(or
(and
(g1 t3o1)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
)
)
)
)
(and
(g1 t3o2)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
)
)
)
)
(and
(g1 t3o3)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
)
)
)
)
(and
(g1 t3o4)
(or
(and
(g2 t3o1)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
)
(and
(g2 t3o2)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
)
(and
(g2 t3o3)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
)
(and
(g2 t3o4)
(and
(or
(g3 t3o1)
(g3 t3o2)
(g3 t3o3)
(g3 t3o4)
)
(and
(=>
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(=>
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(=>
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(=>
(g3 t3o4)
(f t3o4 t3o4 t3o4)
)
)
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
(= q3 false)
)
(or
(and
(g1 t3o1)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o1 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o1 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o1 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o1 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o1 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o1 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o1 t3o4 t3o4)
)
)
)
)
)
(and
(g1 t3o2)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o2 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o2 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o2 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o2 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o2 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o2 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o2 t3o4 t3o4)
)
)
)
)
)
(and
(g1 t3o3)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o3 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o3 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o3 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o3 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o3 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o3 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o3 t3o4 t3o4)
)
)
)
)
)
(and
(g1 t3o4)
(or
(and
(g2 t3o1)
(or
(and
(g3 t3o1)
(f t3o4 t3o1 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o1 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o1 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o1 t3o4)
)
)
)
(and
(g2 t3o2)
(or
(and
(g3 t3o1)
(f t3o4 t3o2 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o2 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o2 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o2 t3o4)
)
)
)
(and
(g2 t3o3)
(or
(and
(g3 t3o1)
(f t3o4 t3o3 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o3 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o3 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o3 t3o4)
)
)
)
(and
(g2 t3o4)
(or
(and
(g3 t3o1)
(f t3o4 t3o4 t3o1)
)
(and
(g3 t3o2)
(f t3o4 t3o4 t3o2)
)
(and
(g3 t3o3)
(f t3o4 t3o4 t3o3)
)
(and
(g3 t3o4)
(f t3o4 t3o4 t3o4)
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
(= c1 true)
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
(g1 c1o3)
(g1 c1o4)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
)
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
(= q2 true)
(= q3 false)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
(g1 c1o3)
(g1 c1o4)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 true)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
(g1 c1o3)
(g1 c1o4)
)
(and
(=>
(g1 c1o1)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o2)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o3)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o4)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 false)
)
(and
(or
(g1 c1o1)
(g1 c1o2)
(g1 c1o3)
(g1 c1o4)
)
(and
(=>
(g1 c1o1)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
)
)
)
(=>
(g1 c1o2)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
)
)
)
(=>
(g1 c1o3)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
)
)
)
(=>
(g1 c1o4)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 true)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
)
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
)
(and
(=>
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
)
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
(= q3 false)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
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
)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 true)
)
(or
(and
(g1 c1o1)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o2)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o3)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o4)
(or
(and
(g2 c1o1)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(or
(g3 c1o1)
(g3 c1o2)
(g3 c1o3)
(g3 c1o4)
)
(and
(=>
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(=>
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(=>
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(=>
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 false)
)
(or
(and
(g1 c1o1)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o1 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o1 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o1 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o1 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o1 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o1 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o1 c1o4 c1o4)
)
)
)
)
)
(and
(g1 c1o2)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o2 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o2 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o2 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o2 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o2 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o2 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o2 c1o4 c1o4)
)
)
)
)
)
(and
(g1 c1o3)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o3 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o3 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o3 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o3 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o3 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o3 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o3 c1o4 c1o4)
)
)
)
)
)
(and
(g1 c1o4)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(f c1o4 c1o1 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o1 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o1 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o1 c1o4)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(f c1o4 c1o2 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o2 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o2 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o2 c1o4)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(f c1o4 c1o3 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o3 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o3 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o3 c1o4)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(f c1o4 c1o4 c1o1)
)
(and
(g3 c1o2)
(f c1o4 c1o4 c1o2)
)
(and
(g3 c1o3)
(f c1o4 c1o4 c1o3)
)
(and
(g3 c1o4)
(f c1o4 c1o4 c1o4)
)
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
(= c1 false)
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
)
(or
(and
(g1 c1o1)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o2)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o3)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o4)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
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
(= q2 true)
(= q3 false)
)
(or
(and
(g1 c1o1)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o2)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o3)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o4)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
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
(= q3 true)
)
(or
(and
(g1 c1o1)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o2)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o3)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o4)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
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
(= q3 false)
)
(or
(and
(g1 c1o1)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o2)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o3)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(and
(g1 c1o4)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c1o1)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o2)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o3)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o4)
(or
(and
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c1o1)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o2)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o3)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o4)
(or
(and
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(and
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(and
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(and
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c1o1)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o2)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o3)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o4)
(and
(=>
(g2 c1o1)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(or
(and
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(and
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(and
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(and
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c1o1)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o1 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o1 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o1 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o1 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o2)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o2 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o2 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o2 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o2 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o3)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o3 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o3 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o3 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o3 c1o4 c1o4)
)
)
)
)
)
)
(=>
(g1 c1o4)
(and
(=>
(g2 c1o1)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o1 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o1 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o1 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o1 c1o4)
)
)
)
)
(=>
(g2 c1o2)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o2 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o2 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o2 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o2 c1o4)
)
)
)
)
(=>
(g2 c1o3)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o3 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o3 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o3 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o3 c1o4)
)
)
)
)
(=>
(g2 c1o4)
(and
(=>
(g3 c1o1)
(not
(f c1o4 c1o4 c1o1)
)
)
(=>
(g3 c1o2)
(not
(f c1o4 c1o4 c1o2)
)
)
(=>
(g3 c1o3)
(not
(f c1o4 c1o4 c1o3)
)
)
(=>
(g3 c1o4)
(not
(f c1o4 c1o4 c1o4)
)
)
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
(= q3 true)
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
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
)
(and
(=>
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
)
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
)
(and
(=>
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
)
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
)
(and
(=>
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
)
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
(= q2 true)
(= q3 false)
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
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
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
)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
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
)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 true)
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
(or
(and
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o2)
(or
(and
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o3)
(or
(and
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 false)
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
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
)
)
)
(=>
(g1 c2o2)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
)
)
)
(=>
(g1 c2o3)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 c2o1)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
)
(and
(=>
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
)
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
)
(and
(=>
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
)
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
)
(and
(=>
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
)
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
(= q3 false)
)
(or
(and
(g1 c2o1)
(and
(or
(g2 c2o1)
(g2 c2o2)
(g2 c2o3)
)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
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
)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
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
)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 c2o1)
(or
(and
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o2)
(or
(and
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o3)
(or
(and
(g2 c2o1)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(or
(g3 c2o1)
(g3 c2o2)
(g3 c2o3)
)
(and
(=>
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(=>
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(=>
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 false)
)
(or
(and
(g1 c2o1)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o1 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o1 c2o3)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o1 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o2 c2o3)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o1 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o1 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o1 c2o3 c2o3)
)
)
)
)
)
(and
(g1 c2o2)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o2 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o1 c2o3)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o2 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o2 c2o3)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o2 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o2 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o2 c2o3 c2o3)
)
)
)
)
)
(and
(g1 c2o3)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(f c2o3 c2o1 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o1 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o1 c2o3)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(f c2o3 c2o2 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o2 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o2 c2o3)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(f c2o3 c2o3 c2o1)
)
(and
(g3 c2o2)
(f c2o3 c2o3 c2o2)
)
(and
(g3 c2o3)
(f c2o3 c2o3 c2o3)
)
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
(= c2 false)
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
)
(or
(and
(g1 c2o1)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o2)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o3)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
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
(= q2 true)
(= q3 false)
)
(or
(and
(g1 c2o1)
(or
(and
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o2)
(or
(and
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o3)
(or
(and
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
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
(= q3 true)
)
(or
(and
(g1 c2o1)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o2)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o3)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
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
(= q3 false)
)
(or
(and
(g1 c2o1)
(and
(=>
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o2)
(and
(=>
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(and
(g1 c2o3)
(and
(=>
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c2o1)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o2)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o3)
(or
(and
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c2o1)
(or
(and
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o2)
(or
(and
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o3)
(or
(and
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(and
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(and
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c2o1)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o2)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o3)
(and
(=>
(g2 c2o1)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(or
(and
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(and
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(and
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c2o1)
(and
(=>
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o1 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o1 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o1 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o2)
(and
(=>
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o2 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o2 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o2 c2o3 c2o3)
)
)
)
)
)
)
(=>
(g1 c2o3)
(and
(=>
(g2 c2o1)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o1 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o1 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o1 c2o3)
)
)
)
)
(=>
(g2 c2o2)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o2 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o2 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o2 c2o3)
)
)
)
)
(=>
(g2 c2o3)
(and
(=>
(g3 c2o1)
(not
(f c2o3 c2o3 c2o1)
)
)
(=>
(g3 c2o2)
(not
(f c2o3 c2o3 c2o2)
)
)
(=>
(g3 c2o3)
(not
(f c2o3 c2o3 c2o3)
)
)
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
(= q3 true)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
)
(and
(=>
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
)
(and
(=>
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
)
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
)
(and
(=>
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
)
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
(= q2 true)
(= q3 false)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
)
(and
(=>
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
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
)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 true)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
)
(and
(=>
(g1 c3o1)
(or
(and
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(=>
(g1 c3o2)
(or
(and
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 false)
)
(and
(or
(g1 c3o1)
(g1 c3o2)
)
(and
(=>
(g1 c3o1)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
)
)
)
(=>
(g1 c3o2)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 true)
)
(or
(and
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
)
(and
(=>
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
)
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
)
(and
(=>
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
)
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
(= q3 false)
)
(or
(and
(g1 c3o1)
(and
(or
(g2 c3o1)
(g2 c3o2)
)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
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
)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 true)
)
(or
(and
(g1 c3o1)
(or
(and
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(and
(g1 c3o2)
(or
(and
(g2 c3o1)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(or
(g3 c3o1)
(g3 c3o2)
)
(and
(=>
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(=>
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 false)
)
(or
(and
(g1 c3o1)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o1 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o1 c3o2)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o1 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o1 c3o2 c3o2)
)
)
)
)
)
(and
(g1 c3o2)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(f c3o2 c3o1 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o1 c3o2)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(f c3o2 c3o2 c3o1)
)
(and
(g3 c3o2)
(f c3o2 c3o2 c3o2)
)
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
(= c3 false)
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
)
(or
(and
(g1 c3o1)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(and
(g1 c3o2)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
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
(= q2 true)
(= q3 false)
)
(or
(and
(g1 c3o1)
(or
(and
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(and
(g1 c3o2)
(or
(and
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
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
(= q3 true)
)
(or
(and
(g1 c3o1)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(and
(g1 c3o2)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
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
(= q3 false)
)
(or
(and
(g1 c3o1)
(and
(=>
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(and
(g1 c3o2)
(and
(=>
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c3o1)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(=>
(g1 c3o2)
(or
(and
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c3o1)
(or
(and
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(=>
(g1 c3o2)
(or
(and
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(and
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c3o1)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(=>
(g1 c3o2)
(and
(=>
(g2 c3o1)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(or
(and
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(and
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c3o1)
(and
(=>
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o1 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o1 c3o2 c3o2)
)
)
)
)
)
)
(=>
(g1 c3o2)
(and
(=>
(g2 c3o1)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o1 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o1 c3o2)
)
)
)
)
(=>
(g2 c3o2)
(and
(=>
(g3 c3o1)
(not
(f c3o2 c3o2 c3o1)
)
)
(=>
(g3 c3o2)
(not
(f c3o2 c3o2 c3o2)
)
)
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
(= q3 true)
)
(and
(or
(g1 c4o1)
(g1 c4o2)
(g1 c4o3)
)
(and
(=>
(g1 c4o1)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
)
(and
(=>
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
)
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
)
(and
(=>
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
)
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
)
(and
(=>
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
)
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
(= q2 true)
(= q3 false)
)
(and
(or
(g1 c4o1)
(g1 c4o2)
(g1 c4o3)
)
(and
(=>
(g1 c4o1)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
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
)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
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
)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 true)
)
(and
(or
(g1 c4o1)
(g1 c4o2)
(g1 c4o3)
)
(and
(=>
(g1 c4o1)
(or
(and
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o2)
(or
(and
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o3)
(or
(and
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 false)
)
(and
(or
(g1 c4o1)
(g1 c4o2)
(g1 c4o3)
)
(and
(=>
(g1 c4o1)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
)
)
)
(=>
(g1 c4o2)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
)
)
)
(=>
(g1 c4o3)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 c4o1)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
)
(and
(=>
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
)
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
)
(and
(=>
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
)
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
)
(and
(=>
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
)
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
(= q3 false)
)
(or
(and
(g1 c4o1)
(and
(or
(g2 c4o1)
(g2 c4o2)
(g2 c4o3)
)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
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
)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
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
)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 true)
)
(or
(and
(g1 c4o1)
(or
(and
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o2)
(or
(and
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o3)
(or
(and
(g2 c4o1)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(or
(g3 c4o1)
(g3 c4o2)
(g3 c4o3)
)
(and
(=>
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(=>
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(=>
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 false)
)
(or
(and
(g1 c4o1)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o1 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o1 c4o3)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o1 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o2 c4o3)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o1 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o1 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o1 c4o3 c4o3)
)
)
)
)
)
(and
(g1 c4o2)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o2 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o1 c4o3)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o2 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o2 c4o3)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o2 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o2 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o2 c4o3 c4o3)
)
)
)
)
)
(and
(g1 c4o3)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(f c4o3 c4o1 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o1 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o1 c4o3)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(f c4o3 c4o2 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o2 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o2 c4o3)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(f c4o3 c4o3 c4o1)
)
(and
(g3 c4o2)
(f c4o3 c4o3 c4o2)
)
(and
(g3 c4o3)
(f c4o3 c4o3 c4o3)
)
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
(= c4 false)
(and
(=>
(and
(= q1 true)
(= q2 true)
(= q3 true)
)
(or
(and
(g1 c4o1)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o2)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o3)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
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
(= q2 true)
(= q3 false)
)
(or
(and
(g1 c4o1)
(or
(and
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o2)
(or
(and
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o3)
(or
(and
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
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
(= q3 true)
)
(or
(and
(g1 c4o1)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o2)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o3)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
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
(= q3 false)
)
(or
(and
(g1 c4o1)
(and
(=>
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o2)
(and
(=>
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(and
(g1 c4o3)
(and
(=>
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c4o1)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o2)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o3)
(or
(and
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c4o1)
(or
(and
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o2)
(or
(and
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o3)
(or
(and
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(and
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(and
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 true)
)
(and
(=>
(g1 c4o1)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o2)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o3)
(and
(=>
(g2 c4o1)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(or
(and
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(and
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(and
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
)
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
(= q3 false)
)
(and
(=>
(g1 c4o1)
(and
(=>
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o1 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o1 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o1 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o2)
(and
(=>
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o2 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o2 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o2 c4o3 c4o3)
)
)
)
)
)
)
(=>
(g1 c4o3)
(and
(=>
(g2 c4o1)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o1 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o1 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o1 c4o3)
)
)
)
)
(=>
(g2 c4o2)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o2 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o2 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o2 c4o3)
)
)
)
)
(=>
(g2 c4o3)
(and
(=>
(g3 c4o1)
(not
(f c4o3 c4o3 c4o1)
)
)
(=>
(g3 c4o2)
(not
(f c4o3 c4o3 c4o2)
)
)
(=>
(g3 c4o3)
(not
(f c4o3 c4o3 c4o3)
)
)
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