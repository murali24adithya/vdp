(set-logic ALL)

(declare-datatypes ((Obj 0))
((
(t1o1)(t1o2)(t1o3)(t1o4)(t1o5)(t1o6)(t1o7)(t1o8)(t2o1)(t2o2)(t2o3)(t3o1)(t3o2)(t3o3)(t3o4)(t3o5)(t3o6)(t3o7)(c1o1)(c1o2)(c1o3)(c1o4)(c1o5)(c1o6)(c2o1)(c2o2)(c2o3)(c2o4)(c2o5)(c3o1)(c3o2)(c3o3)(c3o4)(c3o5)
)))

(declare-datatypes ((Label 0))
((
(book)(tvmonitor)(fork)(bowl)(bed)(dog)(chair)(laptop)(sofa)(cat)
)))

(define-fun labelOf((obj Obj)(lbl Label)) Bool
(or
(and (= obj t1o1) (= lbl book))
(and (= obj t1o2) (= lbl book))
(and (= obj t1o3) (= lbl book))
(and (= obj t1o4) (= lbl book))
(and (= obj t1o5) (= lbl laptop))
(and (= obj t1o6) (= lbl tvmonitor))
(and (= obj t1o7) (= lbl sofa))
(and (= obj t1o8) (= lbl cat))
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

(define-fun leftOf((x Obj)(y Obj)) Bool
(or
(and (= x t1o1) (= y t1o2))
(and (= x t1o1) (= y t1o3))
(and (= x t1o1) (= y t1o7))
(and (= x t1o1) (= y t1o8))
(and (= x t1o2) (= y t1o8))
(and (= x t1o3) (= y t1o8))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o2))
(and (= x t1o4) (= y t1o3))
(and (= x t1o4) (= y t1o5))
(and (= x t1o4) (= y t1o7))
(and (= x t1o4) (= y t1o8))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o3))
(and (= x t1o5) (= y t1o8))
(and (= x t1o6) (= y t1o1))
(and (= x t1o6) (= y t1o2))
(and (= x t1o6) (= y t1o3))
(and (= x t1o6) (= y t1o5))
(and (= x t1o6) (= y t1o7))
(and (= x t1o6) (= y t1o8))
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

(define-fun rightOf((x Obj)(y Obj)) Bool
(or
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o6))
(and (= x t1o2) (= y t1o1))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o6))
(and (= x t1o3) (= y t1o1))
(and (= x t1o3) (= y t1o4))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o6))
(and (= x t1o5) (= y t1o4))
(and (= x t1o5) (= y t1o6))
(and (= x t1o7) (= y t1o1))
(and (= x t1o7) (= y t1o4))
(and (= x t1o7) (= y t1o6))
(and (= x t1o8) (= y t1o1))
(and (= x t1o8) (= y t1o2))
(and (= x t1o8) (= y t1o3))
(and (= x t1o8) (= y t1o4))
(and (= x t1o8) (= y t1o5))
(and (= x t1o8) (= y t1o6))
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

(define-fun above((x Obj)(y Obj)) Bool
(or
(and (= x t1o1) (= y t1o3))
(and (= x t1o1) (= y t1o4))
(and (= x t1o1) (= y t1o5))
(and (= x t1o1) (= y t1o8))
(and (= x t1o2) (= y t1o3))
(and (= x t1o2) (= y t1o4))
(and (= x t1o2) (= y t1o5))
(and (= x t1o2) (= y t1o8))
(and (= x t1o3) (= y t1o4))
(and (= x t1o3) (= y t1o5))
(and (= x t1o3) (= y t1o8))
(and (= x t1o4) (= y t1o8))
(and (= x t1o5) (= y t1o8))
(and (= x t1o6) (= y t1o4))
(and (= x t1o6) (= y t1o8))
(and (= x t3o6) (= y t3o5))
(and (= x t3o7) (= y t3o5))
(and (= x c1o3) (= y c1o4))
(and (= x c2o2) (= y c2o5))
(and (= x c2o3) (= y c2o5))
(and (= x c2o4) (= y c2o5))
(and (= x c3o5) (= y c3o1))

))

(define-fun below((x Obj)(y Obj)) Bool
(or
(and (= x t1o3) (= y t1o1))
(and (= x t1o3) (= y t1o2))
(and (= x t1o4) (= y t1o1))
(and (= x t1o4) (= y t1o2))
(and (= x t1o4) (= y t1o3))
(and (= x t1o4) (= y t1o6))
(and (= x t1o5) (= y t1o1))
(and (= x t1o5) (= y t1o2))
(and (= x t1o5) (= y t1o3))
(and (= x t1o8) (= y t1o1))
(and (= x t1o8) (= y t1o2))
(and (= x t1o8) (= y t1o3))
(and (= x t1o8) (= y t1o4))
(and (= x t1o8) (= y t1o5))
(and (= x t1o8) (= y t1o6))
(and (= x t3o5) (= y t3o6))
(and (= x t3o5) (= y t3o7))
(and (= x c1o4) (= y c1o3))
(and (= x c2o5) (= y c2o2))
(and (= x c2o5) (= y c2o3))
(and (= x c2o5) (= y c2o4))
(and (= x c3o1) (= y c3o5))

))

(define-fun within((x Obj)(y Obj)) Bool
(or
(and (= x t1o2) (= y t1o7))
(and (= x t1o3) (= y t1o7))
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

;;Number of variables: 2. Call them x1 and x2

;;Boolean variables to indicate quantifiers. Call them q1 and q2. 'True' corresponds to universal quantifier.
(declare-const q1 Bool)
(declare-const q2 Bool)

;;Guards to synthesize for each quantifier. Call them g1 (for x1) and g2 (for x2).
(declare-const g1lbl Label)
(define-fun g1 ((x1 Obj)) Bool
(labelOf x1 g1lbl)
)

(declare-const g2lbl Label)
(define-fun g2 ((x2 Obj)) Bool
(labelOf x2 g2lbl)
)


;;Inner formula
(declare-fun f ( Obj Obj) Bool)


;;Boolean variables to indicate candidate. Call them c1, c2, c3. 'True' corresponds to correct candidate.
(declare-const c1 Bool)
(declare-const c2 Bool)
(declare-const c3 Bool)


;;asserts to express that only one candidate can be chosen
(assert (=> c1 (and (not c2) (not c3))))
(assert (=> c2 (and (not c1) (not c3))))
(assert (=> c3 (and (not c1) (not c2))))
(assert c3)


;;asserts for truth/falsehood of formula on each model. Including vacuity.
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
(get-model)