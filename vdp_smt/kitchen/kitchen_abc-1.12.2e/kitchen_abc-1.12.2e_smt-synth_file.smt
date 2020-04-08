(set-logic ALL)

(define-fun t1o1 () (_ BitVec 5) (_ bv0 5))
(define-fun t1o2 () (_ BitVec 5) (_ bv1 5))
(define-fun t1o3 () (_ BitVec 5) (_ bv2 5))
(define-fun t1o4 () (_ BitVec 5) (_ bv3 5))
(define-fun t1o5 () (_ BitVec 5) (_ bv4 5))
(define-fun t1o6 () (_ BitVec 5) (_ bv5 5))
(define-fun t1o7 () (_ BitVec 5) (_ bv6 5))
(define-fun t1o8 () (_ BitVec 5) (_ bv7 5))
(define-fun t1o9 () (_ BitVec 5) (_ bv8 5))
(define-fun t1o10 () (_ BitVec 5) (_ bv9 5))
(define-fun t2o1 () (_ BitVec 5) (_ bv10 5))
(define-fun t2o2 () (_ BitVec 5) (_ bv11 5))
(define-fun t2o3 () (_ BitVec 5) (_ bv12 5))
(define-fun t3o1 () (_ BitVec 5) (_ bv13 5))
(define-fun t3o2 () (_ BitVec 5) (_ bv14 5))
(define-fun t3o3 () (_ BitVec 5) (_ bv15 5))
(define-fun t3o4 () (_ BitVec 5) (_ bv16 5))
(define-fun t3o5 () (_ BitVec 5) (_ bv17 5))
(define-fun c1o1 () (_ BitVec 5) (_ bv18 5))
(define-fun c1o2 () (_ BitVec 5) (_ bv19 5))
(define-fun c1o3 () (_ BitVec 5) (_ bv20 5))
(define-fun c2o1 () (_ BitVec 5) (_ bv21 5))
(define-fun c2o2 () (_ BitVec 5) (_ bv22 5))
(define-fun c3o1 () (_ BitVec 5) (_ bv23 5))
(define-fun c3o2 () (_ BitVec 5) (_ bv24 5))
(define-fun c3o3 () (_ BitVec 5) (_ bv25 5))


(define-fun bottle () (_ BitVec 4) (_ bv0 4))
(define-fun cup () (_ BitVec 4) (_ bv1 4))
(define-fun sofa () (_ BitVec 4) (_ bv2 4))
(define-fun refrigerator () (_ BitVec 4) (_ bv3 4))
(define-fun bowl () (_ BitVec 4) (_ bv4 4))
(define-fun diningtable () (_ BitVec 4) (_ bv5 4))
(define-fun oven () (_ BitVec 4) (_ bv6 4))
(define-fun chair () (_ BitVec 4) (_ bv7 4))
(define-fun spoon () (_ BitVec 4) (_ bv8 4))
(define-fun sink () (_ BitVec 4) (_ bv9 4))


(define-fun labelOf((obj (_ BitVec 5))(lbl (_ BitVec 4))) Bool
(or
(and (= obj t1o1) (= lbl refrigerator))
(and (= obj t1o2) (= lbl sink))
(and (= obj t1o3) (= lbl oven))
(and (= obj t1o4) (= lbl bowl))
(and (= obj t1o5) (= lbl bowl))
(and (= obj t1o6) (= lbl bowl))
(and (= obj t1o7) (= lbl cup))
(and (= obj t1o8) (= lbl spoon))
(and (= obj t1o9) (= lbl cup))
(and (= obj t1o10) (= lbl bottle))
(and (= obj t2o1) (= lbl refrigerator))
(and (= obj t2o2) (= lbl sink))
(and (= obj t2o3) (= lbl oven))
(and (= obj t3o1) (= lbl refrigerator))
(and (= obj t3o2) (= lbl sink))
(and (= obj t3o3) (= lbl oven))
(and (= obj t3o4) (= lbl diningtable))
(and (= obj t3o5) (= lbl chair))
(and (= obj c1o1) (= lbl refrigerator))
(and (= obj c1o2) (= lbl sofa))
(and (= obj c1o3) (= lbl chair))
(and (= obj c2o1) (= lbl sink))
(and (= obj c2o2) (= lbl sink))
(and (= obj c3o1) (= lbl refrigerator))
(and (= obj c3o2) (= lbl sink))
(and (= obj c3o3) (= lbl oven))

))

(define-fun leftOf ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o3))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o6))
(and (= x t1o2) (= y t1o7))
(and (= x t1o2) (= y t1o8))
(and (= x t1o2) (= y t1o9))
(and (= x t1o2) (= y t1o10))
(and (= x t1o3) (= y t1o6))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o3))
(and (= x t1o4) (= y t1o5))
(and (= x t1o4) (= y t1o6))
(and (= x t1o4) (= y t1o7))
(and (= x t1o4) (= y t1o9))
(and (= x t1o4) (= y t1o10))
(and (= x t1o5) (= y t1o1))
(and (= x t1o5) (= y t1o3))
(and (= x t1o5) (= y t1o6))
(and (= x t1o5) (= y t1o9))
(and (= x t1o5) (= y t1o10))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o3))
(and (= x t1o7) (= y t1o6))
(and (= x t1o7) (= y t1o9))
(and (= x t1o7) (= y t1o10))
(and (= x t1o8) (= y t1o1))
(and (= x t1o8) (= y t1o3))
(and (= x t1o8) (= y t1o5))
(and (= x t1o8) (= y t1o6))
(and (= x t1o8) (= y t1o7))
(and (= x t1o8) (= y t1o9))
(and (= x t1o8) (= y t1o10))
(and (= x t1o9) (= y t1o1))
(and (= x t1o9) (= y t1o3))
(and (= x t1o9) (= y t1o6))
(and (= x t1o10) (= y t1o1))
(and (= x t1o10) (= y t1o3))
(and (= x t1o10) (= y t1o6))
(and (= x t2o2) (= y t2o1))
(and (= x t2o3) (= y t2o1))
(and (= x t3o1) (= y t3o2))
(and (= x t3o1) (= y t3o4))
(and (= x t3o1) (= y t3o5))
(and (= x t3o3) (= y t3o2))
(and (= x t3o3) (= y t3o4))
(and (= x t3o3) (= y t3o5))
(and (= x t3o4) (= y t3o2))
(and (= x t3o5) (= y t3o2))
(and (= x c1o1) (= y c1o2))
(and (= x c1o3) (= y c1o2))
(and (= x c3o1) (= y c3o3))
(and (= x c3o2) (= y c3o3))

))

(define-fun rightOf ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o5))
(and (= x t1o1) (= y t1o7))
(and (= x t1o1) (= y t1o8))
(and (= x t1o1) (= y t1o9))
(and (= x t1o1) (= y t1o10))
(and (= x t1o3) (= y t1o2))
(and (= x t1o3) (= y t1o4))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o7))
(and (= x t1o3) (= y t1o8))
(and (= x t1o3) (= y t1o9))
(and (= x t1o3) (= y t1o10))
(and (= x t1o4) (= y t1o2))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o4))
(and (= x t1o5) (= y t1o8))
(and (= x t1o6) (= y t1o2))
(and (= x t1o6) (= y t1o3))
(and (= x t1o6) (= y t1o4))
(and (= x t1o6) (= y t1o5))
(and (= x t1o6) (= y t1o7))
(and (= x t1o6) (= y t1o8))
(and (= x t1o6) (= y t1o9))
(and (= x t1o6) (= y t1o10))
(and (= x t1o7) (= y t1o2))
(and (= x t1o7) (= y t1o4))
(and (= x t1o7) (= y t1o8))
(and (= x t1o8) (= y t1o2))
(and (= x t1o9) (= y t1o2))
(and (= x t1o9) (= y t1o4))
(and (= x t1o9) (= y t1o5))
(and (= x t1o9) (= y t1o7))
(and (= x t1o9) (= y t1o8))
(and (= x t1o10) (= y t1o2))
(and (= x t1o10) (= y t1o4))
(and (= x t1o10) (= y t1o5))
(and (= x t1o10) (= y t1o7))
(and (= x t1o10) (= y t1o8))
(and (= x t2o1) (= y t2o2))
(and (= x t2o1) (= y t2o3))
(and (= x t3o2) (= y t3o1))
(and (= x t3o2) (= y t3o3))
(and (= x t3o2) (= y t3o4))
(and (= x t3o2) (= y t3o5))
(and (= x t3o4) (= y t3o1))
(and (= x t3o4) (= y t3o3))
(and (= x t3o5) (= y t3o1))
(and (= x t3o5) (= y t3o3))
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o3))
(and (= x c3o3) (= y c3o1))
(and (= x c3o3) (= y c3o2))

))

(define-fun above ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o2) (= y t1o1))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o6))
(and (= x t1o5) (= y t1o1))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o6))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o2))
(and (= x t1o7) (= y t1o6))
(and (= x t1o8) (= y t1o1))
(and (= x t1o8) (= y t1o2))
(and (= x t1o8) (= y t1o3))
(and (= x t1o8) (= y t1o4))
(and (= x t1o8) (= y t1o5))
(and (= x t1o8) (= y t1o6))
(and (= x t1o8) (= y t1o7))
(and (= x t1o8) (= y t1o9))
(and (= x t1o9) (= y t1o1))
(and (= x t1o9) (= y t1o2))
(and (= x t1o9) (= y t1o6))
(and (= x t1o10) (= y t1o1))
(and (= x t1o10) (= y t1o2))
(and (= x t1o10) (= y t1o3))
(and (= x t1o10) (= y t1o4))
(and (= x t1o10) (= y t1o5))
(and (= x t1o10) (= y t1o6))
(and (= x t1o10) (= y t1o7))
(and (= x t1o10) (= y t1o9))
(and (= x t2o2) (= y t2o3))
(and (= x t3o4) (= y t3o2))
(and (= x t3o5) (= y t3o2))
(and (= x c2o2) (= y c2o1))

))

(define-fun below ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o5))
(and (= x t1o1) (= y t1o7))
(and (= x t1o1) (= y t1o8))
(and (= x t1o1) (= y t1o9))
(and (= x t1o1) (= y t1o10))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o7))
(and (= x t1o2) (= y t1o8))
(and (= x t1o2) (= y t1o9))
(and (= x t1o2) (= y t1o10))
(and (= x t1o3) (= y t1o8))
(and (= x t1o3) (= y t1o10))
(and (= x t1o4) (= y t1o8))
(and (= x t1o4) (= y t1o10))
(and (= x t1o5) (= y t1o8))
(and (= x t1o5) (= y t1o10))
(and (= x t1o6) (= y t1o4))
(and (= x t1o6) (= y t1o5))
(and (= x t1o6) (= y t1o7))
(and (= x t1o6) (= y t1o8))
(and (= x t1o6) (= y t1o9))
(and (= x t1o6) (= y t1o10))
(and (= x t1o7) (= y t1o8))
(and (= x t1o7) (= y t1o10))
(and (= x t1o9) (= y t1o8))
(and (= x t1o9) (= y t1o10))
(and (= x t2o3) (= y t2o2))
(and (= x t3o2) (= y t3o4))
(and (= x t3o2) (= y t3o5))
(and (= x c2o1) (= y c2o2))

))

(define-fun within ((x (_ BitVec 5))(y (_ BitVec 5))) Bool
(or
(and (= x t3o3) (= y t3o1))

))



(define-fun eq ((x1 (_ BitVec 5))(x2 (_ BitVec 5))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 5))(x2 (_ BitVec 5))) Bool (not (= x1 x2)))


(declare-const q1 Bool)
(declare-const q2 Bool)

(declare-const lbl_g1 (_ BitVec 4))
(assert (or
(= lbl_g1 bottle)
(= lbl_g1 cup)
(= lbl_g1 sofa)
(= lbl_g1 refrigerator)
(= lbl_g1 bowl)
(= lbl_g1 diningtable)
(= lbl_g1 oven)
(= lbl_g1 chair)
(= lbl_g1 spoon)
(= lbl_g1 sink)
)
)
(define-fun g1 ((x (_ BitVec 5))) Bool
(labelOf x lbl_g1)
)

(declare-const lbl_g2 (_ BitVec 4))
(assert (or
(= lbl_g2 bottle)
(= lbl_g2 cup)
(= lbl_g2 sofa)
(= lbl_g2 refrigerator)
(= lbl_g2 bowl)
(= lbl_g2 diningtable)
(= lbl_g2 oven)
(= lbl_g2 chair)
(= lbl_g2 spoon)
(= lbl_g2 sink)
)
)
(define-fun g2 ((x (_ BitVec 5))) Bool
(labelOf x lbl_g2)
)


(declare-const constval_bottle (_ BitVec 4))
(assert (= constval_bottle bottle))
(declare-const constval_cup (_ BitVec 4))
(assert (= constval_cup cup))
(declare-const constval_sofa (_ BitVec 4))
(assert (= constval_sofa sofa))
(declare-const constval_refrigerator (_ BitVec 4))
(assert (= constval_refrigerator refrigerator))
(declare-const constval_bowl (_ BitVec 4))
(assert (= constval_bowl bowl))
(declare-const constval_diningtable (_ BitVec 4))
(assert (= constval_diningtable diningtable))
(declare-const constval_oven (_ BitVec 4))
(assert (= constval_oven oven))
(declare-const constval_chair (_ BitVec 4))
(assert (= constval_chair chair))
(declare-const constval_spoon (_ BitVec 4))
(assert (= constval_spoon spoon))
(declare-const constval_sink (_ BitVec 4))
(assert (= constval_sink sink))

(declare-const eq_x1_x1 Bool)
(declare-const eq_x1_x2 Bool)
(declare-const eq_x2_x1 Bool)
(declare-const eq_x2_x2 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const neq_x1_x2 Bool)
(declare-const neq_x2_x1 Bool)
(declare-const neq_x2_x2 Bool)
(declare-const labelOf_x1_bottle Bool)
(declare-const labelOf_x1_cup Bool)
(declare-const labelOf_x1_sofa Bool)
(declare-const labelOf_x1_refrigerator Bool)
(declare-const labelOf_x1_bowl Bool)
(declare-const labelOf_x1_diningtable Bool)
(declare-const labelOf_x1_oven Bool)
(declare-const labelOf_x1_chair Bool)
(declare-const labelOf_x1_spoon Bool)
(declare-const labelOf_x1_sink Bool)
(declare-const labelOf_x2_bottle Bool)
(declare-const labelOf_x2_cup Bool)
(declare-const labelOf_x2_sofa Bool)
(declare-const labelOf_x2_refrigerator Bool)
(declare-const labelOf_x2_bowl Bool)
(declare-const labelOf_x2_diningtable Bool)
(declare-const labelOf_x2_oven Bool)
(declare-const labelOf_x2_chair Bool)
(declare-const labelOf_x2_spoon Bool)
(declare-const labelOf_x2_sink Bool)
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

(define-fun f ((x1 (_ BitVec 5)) (x2 (_ BitVec 5))) Bool
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
labelOf_x1_bottle
(labelOf x1 bottle)
)
(=>
labelOf_x1_cup
(labelOf x1 cup)
)
(=>
labelOf_x1_sofa
(labelOf x1 sofa)
)
(=>
labelOf_x1_refrigerator
(labelOf x1 refrigerator)
)
(=>
labelOf_x1_bowl
(labelOf x1 bowl)
)
(=>
labelOf_x1_diningtable
(labelOf x1 diningtable)
)
(=>
labelOf_x1_oven
(labelOf x1 oven)
)
(=>
labelOf_x1_chair
(labelOf x1 chair)
)
(=>
labelOf_x1_spoon
(labelOf x1 spoon)
)
(=>
labelOf_x1_sink
(labelOf x1 sink)
)
(=>
labelOf_x2_bottle
(labelOf x2 bottle)
)
(=>
labelOf_x2_cup
(labelOf x2 cup)
)
(=>
labelOf_x2_sofa
(labelOf x2 sofa)
)
(=>
labelOf_x2_refrigerator
(labelOf x2 refrigerator)
)
(=>
labelOf_x2_bowl
(labelOf x2 bowl)
)
(=>
labelOf_x2_diningtable
(labelOf x2 diningtable)
)
(=>
labelOf_x2_oven
(labelOf x2 oven)
)
(=>
labelOf_x2_chair
(labelOf x2 chair)
)
(=>
labelOf_x2_spoon
(labelOf x2 spoon)
)
(=>
labelOf_x2_sink
(labelOf x2 sink)
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
)
(and
(=>
(g1 c1o1)
(and
(or
(g2 c1o1)
(g2 c1o2)
(g2 c1o3)
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


(check-sat)
(get-model)