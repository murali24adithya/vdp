(set-logic ALL)

(define-fun t1o1 () (_ BitVec 6) (_ bv0 6))
(define-fun t1o2 () (_ BitVec 6) (_ bv1 6))
(define-fun t1o3 () (_ BitVec 6) (_ bv2 6))
(define-fun t2o1 () (_ BitVec 6) (_ bv3 6))
(define-fun t2o2 () (_ BitVec 6) (_ bv4 6))
(define-fun t3o1 () (_ BitVec 6) (_ bv5 6))
(define-fun t3o2 () (_ BitVec 6) (_ bv6 6))
(define-fun t3o3 () (_ BitVec 6) (_ bv7 6))
(define-fun t3o4 () (_ BitVec 6) (_ bv8 6))
(define-fun c1o1 () (_ BitVec 6) (_ bv9 6))
(define-fun c1o2 () (_ BitVec 6) (_ bv10 6))
(define-fun c1o3 () (_ BitVec 6) (_ bv11 6))
(define-fun c1o4 () (_ BitVec 6) (_ bv12 6))
(define-fun c1o5 () (_ BitVec 6) (_ bv13 6))
(define-fun c1o6 () (_ BitVec 6) (_ bv14 6))
(define-fun c1o7 () (_ BitVec 6) (_ bv15 6))
(define-fun c1o8 () (_ BitVec 6) (_ bv16 6))
(define-fun c1o9 () (_ BitVec 6) (_ bv17 6))
(define-fun c2o1 () (_ BitVec 6) (_ bv18 6))
(define-fun c2o2 () (_ BitVec 6) (_ bv19 6))
(define-fun c3o1 () (_ BitVec 6) (_ bv20 6))
(define-fun c3o2 () (_ BitVec 6) (_ bv21 6))
(define-fun c3o3 () (_ BitVec 6) (_ bv22 6))
(define-fun c3o4 () (_ BitVec 6) (_ bv23 6))
(define-fun c3o5 () (_ BitVec 6) (_ bv24 6))
(define-fun c3o6 () (_ BitVec 6) (_ bv25 6))
(define-fun c3o7 () (_ BitVec 6) (_ bv26 6))
(define-fun c3o8 () (_ BitVec 6) (_ bv27 6))
(define-fun c3o9 () (_ BitVec 6) (_ bv28 6))
(define-fun c4o1 () (_ BitVec 6) (_ bv29 6))
(define-fun c4o2 () (_ BitVec 6) (_ bv30 6))
(define-fun c4o3 () (_ BitVec 6) (_ bv31 6))
(define-fun c4o4 () (_ BitVec 6) (_ bv32 6))
(define-fun c4o5 () (_ BitVec 6) (_ bv33 6))
(define-fun c4o6 () (_ BitVec 6) (_ bv34 6))
(define-fun c4o7 () (_ BitVec 6) (_ bv35 6))
(define-fun c4o8 () (_ BitVec 6) (_ bv36 6))
(define-fun c4o9 () (_ BitVec 6) (_ bv37 6))


(define-fun chair () (_ BitVec 3) (_ bv0 3))
(define-fun dog () (_ BitVec 3) (_ bv1 3))
(define-fun person () (_ BitVec 3) (_ bv2 3))
(define-fun tie () (_ BitVec 3) (_ bv3 3))
(define-fun cell_phone () (_ BitVec 3) (_ bv4 3))


(define-fun labelOf((obj (_ BitVec 6))(lbl (_ BitVec 3))) Bool
(or
(and (= obj t1o1) (= lbl tie))
(and (= obj t1o2) (= lbl person))
(and (= obj t1o3) (= lbl person))
(and (= obj t2o1) (= lbl tie))
(and (= obj t2o2) (= lbl person))
(and (= obj t3o1) (= lbl tie))
(and (= obj t3o2) (= lbl person))
(and (= obj t3o3) (= lbl person))
(and (= obj t3o4) (= lbl person))
(and (= obj c1o1) (= lbl cell_phone))
(and (= obj c1o2) (= lbl chair))
(and (= obj c1o3) (= lbl person))
(and (= obj c1o4) (= lbl person))
(and (= obj c1o5) (= lbl person))
(and (= obj c1o6) (= lbl person))
(and (= obj c1o7) (= lbl person))
(and (= obj c1o8) (= lbl person))
(and (= obj c1o9) (= lbl person))
(and (= obj c2o1) (= lbl tie))
(and (= obj c2o2) (= lbl dog))
(and (= obj c3o1) (= lbl tie))
(and (= obj c3o2) (= lbl tie))
(and (= obj c3o3) (= lbl tie))
(and (= obj c3o4) (= lbl tie))
(and (= obj c3o5) (= lbl tie))
(and (= obj c3o6) (= lbl tie))
(and (= obj c3o7) (= lbl tie))
(and (= obj c3o8) (= lbl tie))
(and (= obj c3o9) (= lbl person))
(and (= obj c4o1) (= lbl tie))
(and (= obj c4o2) (= lbl tie))
(and (= obj c4o3) (= lbl tie))
(and (= obj c4o4) (= lbl person))
(and (= obj c4o5) (= lbl person))
(and (= obj c4o6) (= lbl person))
(and (= obj c4o7) (= lbl person))
(and (= obj c4o8) (= lbl person))
(and (= obj c4o9) (= lbl person))

))

(define-fun leftOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o3))
(and (= x t3o1) (= y t3o2))
(and (= x t3o3) (= y t3o2))
(and (= x c1o1) (= y c1o2))
(and (= x c1o1) (= y c1o4))
(and (= x c1o1) (= y c1o6))
(and (= x c1o1) (= y c1o8))
(and (= x c1o1) (= y c1o9))
(and (= x c1o3) (= y c1o2))
(and (= x c1o3) (= y c1o4))
(and (= x c1o3) (= y c1o6))
(and (= x c1o3) (= y c1o8))
(and (= x c1o4) (= y c1o2))
(and (= x c1o5) (= y c1o2))
(and (= x c1o5) (= y c1o4))
(and (= x c1o5) (= y c1o6))
(and (= x c1o5) (= y c1o8))
(and (= x c1o5) (= y c1o9))
(and (= x c1o7) (= y c1o2))
(and (= x c1o7) (= y c1o4))
(and (= x c1o7) (= y c1o6))
(and (= x c1o7) (= y c1o8))
(and (= x c1o9) (= y c1o2))
(and (= x c1o9) (= y c1o6))
(and (= x c1o9) (= y c1o8))
(and (= x c3o1) (= y c3o2))
(and (= x c3o1) (= y c3o3))
(and (= x c3o1) (= y c3o5))
(and (= x c3o1) (= y c3o6))
(and (= x c3o1) (= y c3o8))
(and (= x c3o2) (= y c3o6))
(and (= x c3o3) (= y c3o2))
(and (= x c3o3) (= y c3o6))
(and (= x c3o4) (= y c3o1))
(and (= x c3o4) (= y c3o2))
(and (= x c3o4) (= y c3o3))
(and (= x c3o4) (= y c3o5))
(and (= x c3o4) (= y c3o6))
(and (= x c3o4) (= y c3o7))
(and (= x c3o4) (= y c3o8))
(and (= x c3o5) (= y c3o2))
(and (= x c3o5) (= y c3o3))
(and (= x c3o5) (= y c3o6))
(and (= x c3o7) (= y c3o1))
(and (= x c3o7) (= y c3o2))
(and (= x c3o7) (= y c3o3))
(and (= x c3o7) (= y c3o5))
(and (= x c3o7) (= y c3o6))
(and (= x c3o7) (= y c3o8))
(and (= x c3o8) (= y c3o2))
(and (= x c3o8) (= y c3o3))
(and (= x c3o8) (= y c3o5))
(and (= x c3o8) (= y c3o6))
(and (= x c3o9) (= y c3o1))
(and (= x c3o9) (= y c3o2))
(and (= x c3o9) (= y c3o3))
(and (= x c3o9) (= y c3o5))
(and (= x c3o9) (= y c3o6))
(and (= x c3o9) (= y c3o7))
(and (= x c3o9) (= y c3o8))
(and (= x c4o1) (= y c4o2))
(and (= x c4o1) (= y c4o3))
(and (= x c4o1) (= y c4o4))
(and (= x c4o1) (= y c4o6))
(and (= x c4o1) (= y c4o7))
(and (= x c4o1) (= y c4o8))
(and (= x c4o2) (= y c4o3))
(and (= x c4o2) (= y c4o4))
(and (= x c4o2) (= y c4o7))
(and (= x c4o5) (= y c4o2))
(and (= x c4o5) (= y c4o3))
(and (= x c4o5) (= y c4o4))
(and (= x c4o5) (= y c4o6))
(and (= x c4o5) (= y c4o7))
(and (= x c4o6) (= y c4o3))
(and (= x c4o6) (= y c4o4))
(and (= x c4o7) (= y c4o3))
(and (= x c4o8) (= y c4o2))
(and (= x c4o8) (= y c4o3))
(and (= x c4o8) (= y c4o4))
(and (= x c4o8) (= y c4o7))
(and (= x c4o9) (= y c4o1))
(and (= x c4o9) (= y c4o2))
(and (= x c4o9) (= y c4o3))
(and (= x c4o9) (= y c4o4))
(and (= x c4o9) (= y c4o6))
(and (= x c4o9) (= y c4o7))
(and (= x c4o9) (= y c4o8))

))

(define-fun rightOf ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o3) (= y t1o1))
(and (= x t3o2) (= y t3o1))
(and (= x t3o2) (= y t3o3))
(and (= x c1o2) (= y c1o1))
(and (= x c1o2) (= y c1o3))
(and (= x c1o2) (= y c1o4))
(and (= x c1o2) (= y c1o5))
(and (= x c1o2) (= y c1o7))
(and (= x c1o2) (= y c1o9))
(and (= x c1o4) (= y c1o1))
(and (= x c1o4) (= y c1o3))
(and (= x c1o4) (= y c1o5))
(and (= x c1o4) (= y c1o7))
(and (= x c1o6) (= y c1o1))
(and (= x c1o6) (= y c1o3))
(and (= x c1o6) (= y c1o5))
(and (= x c1o6) (= y c1o7))
(and (= x c1o6) (= y c1o9))
(and (= x c1o8) (= y c1o1))
(and (= x c1o8) (= y c1o3))
(and (= x c1o8) (= y c1o5))
(and (= x c1o8) (= y c1o7))
(and (= x c1o8) (= y c1o9))
(and (= x c1o9) (= y c1o1))
(and (= x c1o9) (= y c1o5))
(and (= x c3o1) (= y c3o4))
(and (= x c3o1) (= y c3o7))
(and (= x c3o1) (= y c3o9))
(and (= x c3o2) (= y c3o1))
(and (= x c3o2) (= y c3o3))
(and (= x c3o2) (= y c3o4))
(and (= x c3o2) (= y c3o5))
(and (= x c3o2) (= y c3o7))
(and (= x c3o2) (= y c3o8))
(and (= x c3o2) (= y c3o9))
(and (= x c3o3) (= y c3o1))
(and (= x c3o3) (= y c3o4))
(and (= x c3o3) (= y c3o5))
(and (= x c3o3) (= y c3o7))
(and (= x c3o3) (= y c3o8))
(and (= x c3o3) (= y c3o9))
(and (= x c3o5) (= y c3o1))
(and (= x c3o5) (= y c3o4))
(and (= x c3o5) (= y c3o7))
(and (= x c3o5) (= y c3o8))
(and (= x c3o5) (= y c3o9))
(and (= x c3o6) (= y c3o1))
(and (= x c3o6) (= y c3o2))
(and (= x c3o6) (= y c3o3))
(and (= x c3o6) (= y c3o4))
(and (= x c3o6) (= y c3o5))
(and (= x c3o6) (= y c3o7))
(and (= x c3o6) (= y c3o8))
(and (= x c3o6) (= y c3o9))
(and (= x c3o7) (= y c3o4))
(and (= x c3o7) (= y c3o9))
(and (= x c3o8) (= y c3o1))
(and (= x c3o8) (= y c3o4))
(and (= x c3o8) (= y c3o7))
(and (= x c3o8) (= y c3o9))
(and (= x c4o1) (= y c4o9))
(and (= x c4o2) (= y c4o1))
(and (= x c4o2) (= y c4o5))
(and (= x c4o2) (= y c4o8))
(and (= x c4o2) (= y c4o9))
(and (= x c4o3) (= y c4o1))
(and (= x c4o3) (= y c4o2))
(and (= x c4o3) (= y c4o5))
(and (= x c4o3) (= y c4o6))
(and (= x c4o3) (= y c4o7))
(and (= x c4o3) (= y c4o8))
(and (= x c4o3) (= y c4o9))
(and (= x c4o4) (= y c4o1))
(and (= x c4o4) (= y c4o2))
(and (= x c4o4) (= y c4o5))
(and (= x c4o4) (= y c4o6))
(and (= x c4o4) (= y c4o8))
(and (= x c4o4) (= y c4o9))
(and (= x c4o6) (= y c4o1))
(and (= x c4o6) (= y c4o5))
(and (= x c4o6) (= y c4o9))
(and (= x c4o7) (= y c4o1))
(and (= x c4o7) (= y c4o2))
(and (= x c4o7) (= y c4o5))
(and (= x c4o7) (= y c4o8))
(and (= x c4o7) (= y c4o9))
(and (= x c4o8) (= y c4o1))
(and (= x c4o8) (= y c4o9))

))

(define-fun above ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x c1o9) (= y c1o1))
(and (= x c1o9) (= y c1o2))

))

(define-fun below ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x c1o1) (= y c1o9))
(and (= x c1o2) (= y c1o9))

))

(define-fun within ((x (_ BitVec 6))(y (_ BitVec 6))) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t2o1) (= y t2o2))
(and (= x t3o1) (= y t3o4))
(and (= x c1o1) (= y c1o3))
(and (= x c2o1) (= y c2o2))
(and (= x c4o1) (= y c4o5))
(and (= x c4o2) (= y c4o6))
(and (= x c4o3) (= y c4o4))

))



(define-fun eq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (= x1 x2))
(define-fun neq ((x1 (_ BitVec 6))(x2 (_ BitVec 6))) Bool (not (= x1 x2)))


(declare-const q1 Bool)
(declare-const q2 Bool)

(declare-const lbl_g1 (_ BitVec 3))
(assert (or
(= lbl_g1 chair)
(= lbl_g1 dog)
(= lbl_g1 person)
(= lbl_g1 tie)
(= lbl_g1 cell_phone)
)
)
(define-fun g1 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g1)
)

(declare-const lbl_g2 (_ BitVec 3))
(assert (or
(= lbl_g2 chair)
(= lbl_g2 dog)
(= lbl_g2 person)
(= lbl_g2 tie)
(= lbl_g2 cell_phone)
)
)
(define-fun g2 ((x (_ BitVec 6))) Bool
(labelOf x lbl_g2)
)


(declare-const constval_chair (_ BitVec 3))
(assert (= constval_chair chair))
(declare-const constval_dog (_ BitVec 3))
(assert (= constval_dog dog))
(declare-const constval_person (_ BitVec 3))
(assert (= constval_person person))
(declare-const constval_tie (_ BitVec 3))
(assert (= constval_tie tie))
(declare-const constval_cell_phone (_ BitVec 3))
(assert (= constval_cell_phone cell_phone))

(declare-const eq_x1_x1 Bool)
(declare-const eq_x1_x2 Bool)
(declare-const eq_x2_x1 Bool)
(declare-const eq_x2_x2 Bool)
(declare-const neq_x1_x1 Bool)
(declare-const neq_x1_x2 Bool)
(declare-const neq_x2_x1 Bool)
(declare-const neq_x2_x2 Bool)
(declare-const labelOf_x1_chair Bool)
(declare-const labelOf_x1_dog Bool)
(declare-const labelOf_x1_person Bool)
(declare-const labelOf_x1_tie Bool)
(declare-const labelOf_x1_cell_phone Bool)
(declare-const labelOf_x2_chair Bool)
(declare-const labelOf_x2_dog Bool)
(declare-const labelOf_x2_person Bool)
(declare-const labelOf_x2_tie Bool)
(declare-const labelOf_x2_cell_phone Bool)
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
labelOf_x1_chair
(labelOf x1 chair)
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
labelOf_x1_tie
(labelOf x1 tie)
)
(=>
labelOf_x1_cell_phone
(labelOf x1 cell_phone)
)
(=>
labelOf_x2_chair
(labelOf x2 chair)
)
(=>
labelOf_x2_dog
(labelOf x2 dog)
)
(=>
labelOf_x2_person
(labelOf x2 person)
)
(=>
labelOf_x2_tie
(labelOf x2 tie)
)
(=>
labelOf_x2_cell_phone
(labelOf x2 cell_phone)
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
)
(and
(=>
(g1 t2o1)
(and
(or
(g2 t2o1)
(g2 t2o2)
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
)
)
)
(=>
(g1 t2o2)
(and
(or
(g2 t2o1)
(g2 t2o2)
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
)
)
)
(and
(g1 t2o2)
(and
(or
(g2 t2o1)
(g2 t2o2)
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
(g1 c3o5)
(g1 c3o6)
(g1 c3o7)
(g1 c3o8)
(g1 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o1 c3o6)
)
(=>
(g2 c3o7)
(f c3o1 c3o7)
)
(=>
(g2 c3o8)
(f c3o1 c3o8)
)
(=>
(g2 c3o9)
(f c3o1 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o2 c3o6)
)
(=>
(g2 c3o7)
(f c3o2 c3o7)
)
(=>
(g2 c3o8)
(f c3o2 c3o8)
)
(=>
(g2 c3o9)
(f c3o2 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o3 c3o6)
)
(=>
(g2 c3o7)
(f c3o3 c3o7)
)
(=>
(g2 c3o8)
(f c3o3 c3o8)
)
(=>
(g2 c3o9)
(f c3o3 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o4 c3o6)
)
(=>
(g2 c3o7)
(f c3o4 c3o7)
)
(=>
(g2 c3o8)
(f c3o4 c3o8)
)
(=>
(g2 c3o9)
(f c3o4 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o5 c3o6)
)
(=>
(g2 c3o7)
(f c3o5 c3o7)
)
(=>
(g2 c3o8)
(f c3o5 c3o8)
)
(=>
(g2 c3o9)
(f c3o5 c3o9)
)
)
)
)
(=>
(g1 c3o6)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o6 c3o1)
)
(=>
(g2 c3o2)
(f c3o6 c3o2)
)
(=>
(g2 c3o3)
(f c3o6 c3o3)
)
(=>
(g2 c3o4)
(f c3o6 c3o4)
)
(=>
(g2 c3o5)
(f c3o6 c3o5)
)
(=>
(g2 c3o6)
(f c3o6 c3o6)
)
(=>
(g2 c3o7)
(f c3o6 c3o7)
)
(=>
(g2 c3o8)
(f c3o6 c3o8)
)
(=>
(g2 c3o9)
(f c3o6 c3o9)
)
)
)
)
(=>
(g1 c3o7)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o7 c3o1)
)
(=>
(g2 c3o2)
(f c3o7 c3o2)
)
(=>
(g2 c3o3)
(f c3o7 c3o3)
)
(=>
(g2 c3o4)
(f c3o7 c3o4)
)
(=>
(g2 c3o5)
(f c3o7 c3o5)
)
(=>
(g2 c3o6)
(f c3o7 c3o6)
)
(=>
(g2 c3o7)
(f c3o7 c3o7)
)
(=>
(g2 c3o8)
(f c3o7 c3o8)
)
(=>
(g2 c3o9)
(f c3o7 c3o9)
)
)
)
)
(=>
(g1 c3o8)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o8 c3o1)
)
(=>
(g2 c3o2)
(f c3o8 c3o2)
)
(=>
(g2 c3o3)
(f c3o8 c3o3)
)
(=>
(g2 c3o4)
(f c3o8 c3o4)
)
(=>
(g2 c3o5)
(f c3o8 c3o5)
)
(=>
(g2 c3o6)
(f c3o8 c3o6)
)
(=>
(g2 c3o7)
(f c3o8 c3o7)
)
(=>
(g2 c3o8)
(f c3o8 c3o8)
)
(=>
(g2 c3o9)
(f c3o8 c3o9)
)
)
)
)
(=>
(g1 c3o9)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o9 c3o1)
)
(=>
(g2 c3o2)
(f c3o9 c3o2)
)
(=>
(g2 c3o3)
(f c3o9 c3o3)
)
(=>
(g2 c3o4)
(f c3o9 c3o4)
)
(=>
(g2 c3o5)
(f c3o9 c3o5)
)
(=>
(g2 c3o6)
(f c3o9 c3o6)
)
(=>
(g2 c3o7)
(f c3o9 c3o7)
)
(=>
(g2 c3o8)
(f c3o9 c3o8)
)
(=>
(g2 c3o9)
(f c3o9 c3o9)
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
(g1 c3o6)
(g1 c3o7)
(g1 c3o8)
(g1 c3o9)
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
(and
(g2 c3o6)
(f c3o1 c3o6)
)
(and
(g2 c3o7)
(f c3o1 c3o7)
)
(and
(g2 c3o8)
(f c3o1 c3o8)
)
(and
(g2 c3o9)
(f c3o1 c3o9)
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
(and
(g2 c3o6)
(f c3o2 c3o6)
)
(and
(g2 c3o7)
(f c3o2 c3o7)
)
(and
(g2 c3o8)
(f c3o2 c3o8)
)
(and
(g2 c3o9)
(f c3o2 c3o9)
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
(and
(g2 c3o6)
(f c3o3 c3o6)
)
(and
(g2 c3o7)
(f c3o3 c3o7)
)
(and
(g2 c3o8)
(f c3o3 c3o8)
)
(and
(g2 c3o9)
(f c3o3 c3o9)
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
(and
(g2 c3o6)
(f c3o4 c3o6)
)
(and
(g2 c3o7)
(f c3o4 c3o7)
)
(and
(g2 c3o8)
(f c3o4 c3o8)
)
(and
(g2 c3o9)
(f c3o4 c3o9)
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
(and
(g2 c3o6)
(f c3o5 c3o6)
)
(and
(g2 c3o7)
(f c3o5 c3o7)
)
(and
(g2 c3o8)
(f c3o5 c3o8)
)
(and
(g2 c3o9)
(f c3o5 c3o9)
)
)
)
(=>
(g1 c3o6)
(or
(and
(g2 c3o1)
(f c3o6 c3o1)
)
(and
(g2 c3o2)
(f c3o6 c3o2)
)
(and
(g2 c3o3)
(f c3o6 c3o3)
)
(and
(g2 c3o4)
(f c3o6 c3o4)
)
(and
(g2 c3o5)
(f c3o6 c3o5)
)
(and
(g2 c3o6)
(f c3o6 c3o6)
)
(and
(g2 c3o7)
(f c3o6 c3o7)
)
(and
(g2 c3o8)
(f c3o6 c3o8)
)
(and
(g2 c3o9)
(f c3o6 c3o9)
)
)
)
(=>
(g1 c3o7)
(or
(and
(g2 c3o1)
(f c3o7 c3o1)
)
(and
(g2 c3o2)
(f c3o7 c3o2)
)
(and
(g2 c3o3)
(f c3o7 c3o3)
)
(and
(g2 c3o4)
(f c3o7 c3o4)
)
(and
(g2 c3o5)
(f c3o7 c3o5)
)
(and
(g2 c3o6)
(f c3o7 c3o6)
)
(and
(g2 c3o7)
(f c3o7 c3o7)
)
(and
(g2 c3o8)
(f c3o7 c3o8)
)
(and
(g2 c3o9)
(f c3o7 c3o9)
)
)
)
(=>
(g1 c3o8)
(or
(and
(g2 c3o1)
(f c3o8 c3o1)
)
(and
(g2 c3o2)
(f c3o8 c3o2)
)
(and
(g2 c3o3)
(f c3o8 c3o3)
)
(and
(g2 c3o4)
(f c3o8 c3o4)
)
(and
(g2 c3o5)
(f c3o8 c3o5)
)
(and
(g2 c3o6)
(f c3o8 c3o6)
)
(and
(g2 c3o7)
(f c3o8 c3o7)
)
(and
(g2 c3o8)
(f c3o8 c3o8)
)
(and
(g2 c3o9)
(f c3o8 c3o9)
)
)
)
(=>
(g1 c3o9)
(or
(and
(g2 c3o1)
(f c3o9 c3o1)
)
(and
(g2 c3o2)
(f c3o9 c3o2)
)
(and
(g2 c3o3)
(f c3o9 c3o3)
)
(and
(g2 c3o4)
(f c3o9 c3o4)
)
(and
(g2 c3o5)
(f c3o9 c3o5)
)
(and
(g2 c3o6)
(f c3o9 c3o6)
)
(and
(g2 c3o7)
(f c3o9 c3o7)
)
(and
(g2 c3o8)
(f c3o9 c3o8)
)
(and
(g2 c3o9)
(f c3o9 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o1 c3o6)
)
(=>
(g2 c3o7)
(f c3o1 c3o7)
)
(=>
(g2 c3o8)
(f c3o1 c3o8)
)
(=>
(g2 c3o9)
(f c3o1 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o2 c3o6)
)
(=>
(g2 c3o7)
(f c3o2 c3o7)
)
(=>
(g2 c3o8)
(f c3o2 c3o8)
)
(=>
(g2 c3o9)
(f c3o2 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o3 c3o6)
)
(=>
(g2 c3o7)
(f c3o3 c3o7)
)
(=>
(g2 c3o8)
(f c3o3 c3o8)
)
(=>
(g2 c3o9)
(f c3o3 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o4 c3o6)
)
(=>
(g2 c3o7)
(f c3o4 c3o7)
)
(=>
(g2 c3o8)
(f c3o4 c3o8)
)
(=>
(g2 c3o9)
(f c3o4 c3o9)
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
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
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
(=>
(g2 c3o6)
(f c3o5 c3o6)
)
(=>
(g2 c3o7)
(f c3o5 c3o7)
)
(=>
(g2 c3o8)
(f c3o5 c3o8)
)
(=>
(g2 c3o9)
(f c3o5 c3o9)
)
)
)
)
(and
(g1 c3o6)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o6 c3o1)
)
(=>
(g2 c3o2)
(f c3o6 c3o2)
)
(=>
(g2 c3o3)
(f c3o6 c3o3)
)
(=>
(g2 c3o4)
(f c3o6 c3o4)
)
(=>
(g2 c3o5)
(f c3o6 c3o5)
)
(=>
(g2 c3o6)
(f c3o6 c3o6)
)
(=>
(g2 c3o7)
(f c3o6 c3o7)
)
(=>
(g2 c3o8)
(f c3o6 c3o8)
)
(=>
(g2 c3o9)
(f c3o6 c3o9)
)
)
)
)
(and
(g1 c3o7)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o7 c3o1)
)
(=>
(g2 c3o2)
(f c3o7 c3o2)
)
(=>
(g2 c3o3)
(f c3o7 c3o3)
)
(=>
(g2 c3o4)
(f c3o7 c3o4)
)
(=>
(g2 c3o5)
(f c3o7 c3o5)
)
(=>
(g2 c3o6)
(f c3o7 c3o6)
)
(=>
(g2 c3o7)
(f c3o7 c3o7)
)
(=>
(g2 c3o8)
(f c3o7 c3o8)
)
(=>
(g2 c3o9)
(f c3o7 c3o9)
)
)
)
)
(and
(g1 c3o8)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o8 c3o1)
)
(=>
(g2 c3o2)
(f c3o8 c3o2)
)
(=>
(g2 c3o3)
(f c3o8 c3o3)
)
(=>
(g2 c3o4)
(f c3o8 c3o4)
)
(=>
(g2 c3o5)
(f c3o8 c3o5)
)
(=>
(g2 c3o6)
(f c3o8 c3o6)
)
(=>
(g2 c3o7)
(f c3o8 c3o7)
)
(=>
(g2 c3o8)
(f c3o8 c3o8)
)
(=>
(g2 c3o9)
(f c3o8 c3o9)
)
)
)
)
(and
(g1 c3o9)
(and
(or
(g2 c3o1)
(g2 c3o2)
(g2 c3o3)
(g2 c3o4)
(g2 c3o5)
(g2 c3o6)
(g2 c3o7)
(g2 c3o8)
(g2 c3o9)
)
(and
(=>
(g2 c3o1)
(f c3o9 c3o1)
)
(=>
(g2 c3o2)
(f c3o9 c3o2)
)
(=>
(g2 c3o3)
(f c3o9 c3o3)
)
(=>
(g2 c3o4)
(f c3o9 c3o4)
)
(=>
(g2 c3o5)
(f c3o9 c3o5)
)
(=>
(g2 c3o6)
(f c3o9 c3o6)
)
(=>
(g2 c3o7)
(f c3o9 c3o7)
)
(=>
(g2 c3o8)
(f c3o9 c3o8)
)
(=>
(g2 c3o9)
(f c3o9 c3o9)
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
(and
(g2 c3o6)
(f c3o1 c3o6)
)
(and
(g2 c3o7)
(f c3o1 c3o7)
)
(and
(g2 c3o8)
(f c3o1 c3o8)
)
(and
(g2 c3o9)
(f c3o1 c3o9)
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
(and
(g2 c3o6)
(f c3o2 c3o6)
)
(and
(g2 c3o7)
(f c3o2 c3o7)
)
(and
(g2 c3o8)
(f c3o2 c3o8)
)
(and
(g2 c3o9)
(f c3o2 c3o9)
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
(and
(g2 c3o6)
(f c3o3 c3o6)
)
(and
(g2 c3o7)
(f c3o3 c3o7)
)
(and
(g2 c3o8)
(f c3o3 c3o8)
)
(and
(g2 c3o9)
(f c3o3 c3o9)
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
(and
(g2 c3o6)
(f c3o4 c3o6)
)
(and
(g2 c3o7)
(f c3o4 c3o7)
)
(and
(g2 c3o8)
(f c3o4 c3o8)
)
(and
(g2 c3o9)
(f c3o4 c3o9)
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
(and
(g2 c3o6)
(f c3o5 c3o6)
)
(and
(g2 c3o7)
(f c3o5 c3o7)
)
(and
(g2 c3o8)
(f c3o5 c3o8)
)
(and
(g2 c3o9)
(f c3o5 c3o9)
)
)
)
(and
(g1 c3o6)
(or
(and
(g2 c3o1)
(f c3o6 c3o1)
)
(and
(g2 c3o2)
(f c3o6 c3o2)
)
(and
(g2 c3o3)
(f c3o6 c3o3)
)
(and
(g2 c3o4)
(f c3o6 c3o4)
)
(and
(g2 c3o5)
(f c3o6 c3o5)
)
(and
(g2 c3o6)
(f c3o6 c3o6)
)
(and
(g2 c3o7)
(f c3o6 c3o7)
)
(and
(g2 c3o8)
(f c3o6 c3o8)
)
(and
(g2 c3o9)
(f c3o6 c3o9)
)
)
)
(and
(g1 c3o7)
(or
(and
(g2 c3o1)
(f c3o7 c3o1)
)
(and
(g2 c3o2)
(f c3o7 c3o2)
)
(and
(g2 c3o3)
(f c3o7 c3o3)
)
(and
(g2 c3o4)
(f c3o7 c3o4)
)
(and
(g2 c3o5)
(f c3o7 c3o5)
)
(and
(g2 c3o6)
(f c3o7 c3o6)
)
(and
(g2 c3o7)
(f c3o7 c3o7)
)
(and
(g2 c3o8)
(f c3o7 c3o8)
)
(and
(g2 c3o9)
(f c3o7 c3o9)
)
)
)
(and
(g1 c3o8)
(or
(and
(g2 c3o1)
(f c3o8 c3o1)
)
(and
(g2 c3o2)
(f c3o8 c3o2)
)
(and
(g2 c3o3)
(f c3o8 c3o3)
)
(and
(g2 c3o4)
(f c3o8 c3o4)
)
(and
(g2 c3o5)
(f c3o8 c3o5)
)
(and
(g2 c3o6)
(f c3o8 c3o6)
)
(and
(g2 c3o7)
(f c3o8 c3o7)
)
(and
(g2 c3o8)
(f c3o8 c3o8)
)
(and
(g2 c3o9)
(f c3o8 c3o9)
)
)
)
(and
(g1 c3o9)
(or
(and
(g2 c3o1)
(f c3o9 c3o1)
)
(and
(g2 c3o2)
(f c3o9 c3o2)
)
(and
(g2 c3o3)
(f c3o9 c3o3)
)
(and
(g2 c3o4)
(f c3o9 c3o4)
)
(and
(g2 c3o5)
(f c3o9 c3o5)
)
(and
(g2 c3o6)
(f c3o9 c3o6)
)
(and
(g2 c3o7)
(f c3o9 c3o7)
)
(and
(g2 c3o8)
(f c3o9 c3o8)
)
(and
(g2 c3o9)
(f c3o9 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o1 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o1 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o1 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o1 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o2 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o2 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o2 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o2 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o3 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o3 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o3 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o3 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o4 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o4 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o4 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o4 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o5 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o5 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o5 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o5 c3o9)
)
)
)
)
(and
(g1 c3o6)
(or
(and
(g2 c3o1)
(not
(f c3o6 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o6 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o6 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o6 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o6 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o6 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o6 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o6 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o6 c3o9)
)
)
)
)
(and
(g1 c3o7)
(or
(and
(g2 c3o1)
(not
(f c3o7 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o7 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o7 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o7 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o7 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o7 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o7 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o7 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o7 c3o9)
)
)
)
)
(and
(g1 c3o8)
(or
(and
(g2 c3o1)
(not
(f c3o8 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o8 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o8 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o8 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o8 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o8 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o8 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o8 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o8 c3o9)
)
)
)
)
(and
(g1 c3o9)
(or
(and
(g2 c3o1)
(not
(f c3o9 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o9 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o9 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o9 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o9 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o9 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o9 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o9 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o9 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o1 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o1 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o1 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o1 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o2 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o2 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o2 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o2 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o3 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o3 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o3 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o3 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o4 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o4 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o4 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o4 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o5 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o5 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o5 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o5 c3o9)
)
)
)
)
(and
(g1 c3o6)
(and
(=>
(g2 c3o1)
(not
(f c3o6 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o6 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o6 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o6 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o6 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o6 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o6 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o6 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o6 c3o9)
)
)
)
)
(and
(g1 c3o7)
(and
(=>
(g2 c3o1)
(not
(f c3o7 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o7 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o7 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o7 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o7 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o7 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o7 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o7 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o7 c3o9)
)
)
)
)
(and
(g1 c3o8)
(and
(=>
(g2 c3o1)
(not
(f c3o8 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o8 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o8 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o8 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o8 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o8 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o8 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o8 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o8 c3o9)
)
)
)
)
(and
(g1 c3o9)
(and
(=>
(g2 c3o1)
(not
(f c3o9 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o9 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o9 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o9 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o9 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o9 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o9 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o9 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o9 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o1 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o1 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o1 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o1 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o2 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o2 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o2 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o2 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o3 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o3 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o3 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o3 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o4 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o4 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o4 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o4 c3o9)
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
(and
(g2 c3o6)
(not
(f c3o5 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o5 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o5 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o5 c3o9)
)
)
)
)
(=>
(g1 c3o6)
(or
(and
(g2 c3o1)
(not
(f c3o6 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o6 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o6 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o6 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o6 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o6 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o6 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o6 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o6 c3o9)
)
)
)
)
(=>
(g1 c3o7)
(or
(and
(g2 c3o1)
(not
(f c3o7 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o7 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o7 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o7 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o7 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o7 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o7 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o7 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o7 c3o9)
)
)
)
)
(=>
(g1 c3o8)
(or
(and
(g2 c3o1)
(not
(f c3o8 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o8 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o8 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o8 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o8 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o8 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o8 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o8 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o8 c3o9)
)
)
)
)
(=>
(g1 c3o9)
(or
(and
(g2 c3o1)
(not
(f c3o9 c3o1)
)
)
(and
(g2 c3o2)
(not
(f c3o9 c3o2)
)
)
(and
(g2 c3o3)
(not
(f c3o9 c3o3)
)
)
(and
(g2 c3o4)
(not
(f c3o9 c3o4)
)
)
(and
(g2 c3o5)
(not
(f c3o9 c3o5)
)
)
(and
(g2 c3o6)
(not
(f c3o9 c3o6)
)
)
(and
(g2 c3o7)
(not
(f c3o9 c3o7)
)
)
(and
(g2 c3o8)
(not
(f c3o9 c3o8)
)
)
(and
(g2 c3o9)
(not
(f c3o9 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o1 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o1 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o1 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o1 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o2 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o2 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o2 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o2 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o3 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o3 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o3 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o3 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o4 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o4 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o4 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o4 c3o9)
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
(=>
(g2 c3o6)
(not
(f c3o5 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o5 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o5 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o5 c3o9)
)
)
)
)
(=>
(g1 c3o6)
(and
(=>
(g2 c3o1)
(not
(f c3o6 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o6 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o6 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o6 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o6 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o6 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o6 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o6 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o6 c3o9)
)
)
)
)
(=>
(g1 c3o7)
(and
(=>
(g2 c3o1)
(not
(f c3o7 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o7 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o7 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o7 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o7 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o7 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o7 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o7 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o7 c3o9)
)
)
)
)
(=>
(g1 c3o8)
(and
(=>
(g2 c3o1)
(not
(f c3o8 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o8 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o8 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o8 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o8 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o8 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o8 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o8 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o8 c3o9)
)
)
)
)
(=>
(g1 c3o9)
(and
(=>
(g2 c3o1)
(not
(f c3o9 c3o1)
)
)
(=>
(g2 c3o2)
(not
(f c3o9 c3o2)
)
)
(=>
(g2 c3o3)
(not
(f c3o9 c3o3)
)
)
(=>
(g2 c3o4)
(not
(f c3o9 c3o4)
)
)
(=>
(g2 c3o5)
(not
(f c3o9 c3o5)
)
)
(=>
(g2 c3o6)
(not
(f c3o9 c3o6)
)
)
(=>
(g2 c3o7)
(not
(f c3o9 c3o7)
)
)
(=>
(g2 c3o8)
(not
(f c3o9 c3o8)
)
)
(=>
(g2 c3o9)
(not
(f c3o9 c3o9)
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
)
)
)
)
)
)
)


(check-sat)
(get-model)