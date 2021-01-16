;Datatype of images
(declare-datatypes () ((Img
 s3 s2 s1 t2 t3 t1 s4
)))
;Constant symbols for objects
;All values are distinct by default
(declare-datatypes () ((Obj
 s3_o1 s3_o2
 s2_o2 s2_o3 s2_o1
 s1_o3 s1_o2 s1_o1 s1_o4
 t2_o1 t2_o3 t2_o2
 t3_o2 t3_o3 t3_o1
 t1_o1 t1_o2 t1_o3
 s4_o1 s4_o2 s4_o3
;extraneous objects
 nullobj chair sofa cat
)))
;Membership of objects in images
;Extraneous objects belong to no images
(define-fun inImg ((x Obj )(y Img)) Bool
(or
(and (= x s3_o1) (= y s3))
(and (= x s3_o2) (= y s3))
(and (= x s2_o2) (= y s2))
(and (= x s2_o3) (= y s2))
(and (= x s2_o1) (= y s2))
(and (= x s1_o3) (= y s1))
(and (= x s1_o2) (= y s1))
(and (= x s1_o1) (= y s1))
(and (= x s1_o4) (= y s1))
(and (= x t2_o1) (= y t2))
(and (= x t2_o3) (= y t2))
(and (= x t2_o2) (= y t2))
(and (= x t3_o2) (= y t3))
(and (= x t3_o3) (= y t3))
(and (= x t3_o1) (= y t3))
(and (= x t1_o1) (= y t1))
(and (= x t1_o2) (= y t1))
(and (= x t1_o3) (= y t1))
(and (= x s4_o1) (= y s4))
(and (= x s4_o2) (= y s4))
(and (= x s4_o3) (= y s4))
))
;Datatype of relations--made into a uniform arity
(declare-datatypes () ((Rels neq labelOf sameLabel toLeft toRight above below within)))
;Relation evaluator
;All relations are made into a uniform arity using 'nullobj' as a default argument
(define-fun re ((r Rels)(a Obj)(b Obj)) Bool
(or
(and (= r neq) (not (= b a)))
(and (= r labelOf) (= a s3_o1) (= b sofa))
(and (= r labelOf) (= a s3_o2) (= b cat))
(and (= r labelOf) (= a s2_o2) (= b cat))
(and (= r labelOf) (= a s2_o3) (= b cat))
(and (= r labelOf) (= a s2_o1) (= b sofa))
(and (= r labelOf) (= a s1_o3) (= b cat))
(and (= r labelOf) (= a s1_o2) (= b chair))
(and (= r labelOf) (= a s1_o1) (= b chair))
(and (= r labelOf) (= a s1_o1) (= b sofa))
(and (= r labelOf) (= a s1_o4) (= b cat))
(and (= r labelOf) (= a s4_o1) (= b sofa))
(and (= r labelOf) (= a s4_o2) (= b cat))
(and (= r labelOf) (= a s4_o3) (= b cat))
(and (= r labelOf) (= a t2_o1) (= b sofa))
(and (= r labelOf) (= a t2_o3) (= b cat))
(and (= r labelOf) (= a t2_o2) (= b cat))
(and (= r labelOf) (= a t3_o2) (= b cat))
(and (= r labelOf) (= a t3_o3) (= b cat))
(and (= r labelOf) (= a t3_o1) (= b sofa))
(and (= r labelOf) (= a t1_o1) (= b sofa))
(and (= r labelOf) (= a t1_o2) (= b cat))
(and (= r labelOf) (= a t1_o3) (= b cat))

(and (= r sameLabel) (= a s3_o1) (= b s3_o1))
(and (= r sameLabel) (= a s3_o2) (= b s3_o2))
(and (= r sameLabel) (= a s2_o2) (= b s2_o2))
(and (= r sameLabel) (= a s2_o2) (= b s2_o3))
(and (= r sameLabel) (= a s2_o3) (= b s2_o2))
(and (= r sameLabel) (= a s2_o3) (= b s2_o3))
(and (= r sameLabel) (= a s2_o1) (= b s2_o1))
(and (= r sameLabel) (= a s1_o3) (= b s1_o3))
(and (= r sameLabel) (= a s1_o3) (= b s1_o4))
(and (= r sameLabel) (= a s1_o2) (= b s1_o2))
(and (= r sameLabel) (= a s1_o2) (= b s1_o1))
(and (= r sameLabel) (= a s1_o1) (= b s1_o2))
(and (= r sameLabel) (= a s1_o1) (= b s1_o1))
(and (= r sameLabel) (= a s1_o4) (= b s1_o3))
(and (= r sameLabel) (= a s1_o4) (= b s1_o4))
(and (= r sameLabel) (= a s4_o1) (= b s4_o1))
(and (= r sameLabel) (= a s4_o2) (= b s4_o2))
(and (= r sameLabel) (= a s4_o2) (= b s4_o3))
(and (= r sameLabel) (= a s4_o3) (= b s4_o2))
(and (= r sameLabel) (= a s4_o3) (= b s4_o3))
(and (= r sameLabel) (= a t2_o1) (= b t2_o1))
(and (= r sameLabel) (= a t2_o3) (= b t2_o3))
(and (= r sameLabel) (= a t2_o3) (= b t2_o2))
(and (= r sameLabel) (= a t2_o2) (= b t2_o3))
(and (= r sameLabel) (= a t2_o2) (= b t2_o2))
(and (= r sameLabel) (= a t3_o2) (= b t3_o2))
(and (= r sameLabel) (= a t3_o2) (= b t3_o3))
(and (= r sameLabel) (= a t3_o3) (= b t3_o2))
(and (= r sameLabel) (= a t3_o3) (= b t3_o3))
(and (= r sameLabel) (= a t3_o1) (= b t3_o1))
(and (= r sameLabel) (= a t1_o1) (= b t1_o1))
(and (= r sameLabel) (= a t1_o2) (= b t1_o2))
(and (= r sameLabel) (= a t1_o2) (= b t1_o3))
(and (= r sameLabel) (= a t1_o3) (= b t1_o2))
(and (= r sameLabel) (= a t1_o3) (= b t1_o3))

(and (= r toLeft) (= a s2_o2) (= b s2_o3))
(and (= r toLeft) (= a s2_o2) (= b s2_o1))
(and (= r toLeft) (= a s1_o3) (= b s1_o1))
(and (= r toLeft) (= a s1_o3) (= b s1_o4))
(and (= r toLeft) (= a s1_o2) (= b s1_o1))
(and (= r toLeft) (= a s1_o2) (= b s1_o4))
(and (= r toLeft) (= a s4_o2) (= b s4_o3))
(and (= r toLeft) (= a t3_o2) (= b t3_o3))
(and (= r toLeft) (= a t1_o3) (= b t1_o2))

(and (= r toRight) (= a s2_o3) (= b s2_o2))
(and (= r toRight) (= a s2_o1) (= b s2_o2))
(and (= r toRight) (= a s1_o1) (= b s1_o3))
(and (= r toRight) (= a s1_o1) (= b s1_o2))
(and (= r toRight) (= a s1_o4) (= b s1_o3))
(and (= r toRight) (= a s1_o4) (= b s1_o2))
(and (= r toRight) (= a s4_o3) (= b s4_o2))
(and (= r toRight) (= a t3_o3) (= b t3_o2))
(and (= r toRight) (= a t1_o2) (= b t1_o3))



(and (= r within) (= a s3_o2) (= b s3_o1))
(and (= r within) (= a s2_o3) (= b s2_o1))
(and (= r within) (= a s1_o3) (= b s1_o2))
(and (= r within) (= a s1_o4) (= b s1_o1))
(and (= r within) (= a s4_o2) (= b s4_o1))
(and (= r within) (= a s4_o3) (= b s4_o1))
(and (= r within) (= a t2_o3) (= b t2_o1))
(and (= r within) (= a t2_o2) (= b t2_o1))
(and (= r within) (= a t3_o2) (= b t3_o1))
(and (= r within) (= a t3_o3) (= b t3_o1))
(and (= r within) (= a t1_o2) (= b t1_o1))
(and (= r within) (= a t1_o3) (= b t1_o1))

))
;Datatype for indicating which quantifier to use in formula
(declare-datatypes () ((Quantifier all one)))
;Datatype of indirections to indicate what the arguments of relations in the abstract baseformula are-- i.e, a datatype of bindings 
;Constants/extraneous objects have bindings as well
(declare-datatypes () ((Binding
 bind_x1 bind_x2 bind_x3
;bindings for extraneous objects
 bind_nullobj bind_chair bind_sofa bind_cat
)))
;Defines how bindings correspond to valuations
;Default lookup is nullobj, but it is used since the function is defined for all bindings
(define-fun bindlook ((x1 Obj)(x2 Obj)(x3 Obj)(bind Binding)) Obj
(ite (= bind bind_x3) x3
(ite (= bind bind_x2) x2
(ite (= bind bind_x1) x1
;Bindings for extraneous objects
(ite (= bind bind_cat) cat
(ite (= bind bind_sofa) sofa
(ite (= bind bind_chair) chair
(ite (= bind bind_nullobj) nullobj
 nullobj)))))))
)
;Defines symbols used to indicate which boolean operators will be used, including negations, and their meanings as well
(declare-datatypes () ((Ops opand opor opimpl)))
(declare-datatypes () ((IfNeg yes no)))
(define-fun opeval ((op Ops) (arg1 Bool) (arg2 Bool)) Bool
 (ite (= op opand) (and arg1 arg2)
 (ite (= op opor) (or arg1 arg2)
  (=> arg1 arg2)
)))
(define-fun negeval ((val IfNeg)(arg Bool)) Bool
 (ite (= val yes)
  (not arg)
  arg
))
(declare-const q1 Quantifier)
(declare-const q2 Quantifier)
(declare-const q3 Quantifier)

(declare-const op1 Ops)
(declare-const op2 Ops)

(declare-const n1 IfNeg)
(declare-const n2 IfNeg)
(declare-const n3 IfNeg)

(declare-const r1 Rels)
(declare-const r2 Rels)
(declare-const r3 Rels)

(declare-const arg1_1 Binding)
(declare-const arg1_2 Binding)
(declare-const arg2_1 Binding)
(declare-const arg2_2 Binding)
(declare-const arg3_1 Binding)
(declare-const arg3_2 Binding)

(assert (= q1 one))
(assert (= q2 one))
(assert (= q3 one))

(assert (= op1 opand))
(assert (= op2 opand))

(assert (=> (= r1 labelOf) (= n1 no)))
(assert (=> (= r2 labelOf) (= n2 no)))
(assert (=> (= r3 labelOf) (= n3 no)))

;Defines the innermost (quantifier-free) formula
(define-fun baseformula ((x1 Obj)(x2 Obj)(x3 Obj)) Bool
(opeval op1
(negeval n1 (re r1 (bindlook x1 x2 x3 arg1_1) (bindlook x1 x2 x3 arg1_2)))
(opeval op2
(negeval n2 (re r2 (bindlook x1 x2 x3 arg2_1) (bindlook x1 x2 x3 arg2_2)))
(negeval n3 (re r3 (bindlook x1 x2 x3 arg3_1) (bindlook x1 x2 x3 arg3_2)))))
)
;Formula at quantifier level 3
(define-fun formula_level_3 ((x1 Obj) (x2 Obj) (img Img)) Bool
(ite (= q3 all)
 (forall ((x3 Obj)) (=> (inImg x3 img) (baseformula x1 x2 x3)))
 (exists ((x3 Obj)) (and (inImg x3 img) (baseformula x1 x2 x3)))
));Formula at quantifier level 2
(define-fun formula_level_2 ((x1 Obj) (img Img)) Bool
(ite (= q2 all)
 (forall ((x2 Obj)) (=> (inImg x2 img) (formula_level_3 x1 x2 img)))
 (exists ((x2 Obj)) (and (inImg x2 img) (formula_level_3 x1 x2 img)))
));Formula at quantifier level 1
(define-fun formula_level_1 ((img Img)) Bool
(ite (= q1 all)
 (forall ((x1 Obj)) (=> (inImg x1 img) (formula_level_2 x1 img)))
 (exists ((x1 Obj)) (and (inImg x1 img) (formula_level_2 x1 img)))
))

(assert (formula_level_1 t2))
(assert (formula_level_1 t3))
(assert (formula_level_1 t1))

(assert (formula_level_1 s4))
(assert (not (formula_level_1 s1)))
(assert (not (formula_level_1 s2)))
(assert (not (formula_level_1 s3)))

(assert (not (and
(= q1 one)
(= q2 one)
(= q3 one)
(= op1 opand)
(= op2 opand)
(= r1 within)
(= n1 no)
(= arg1_1 bind_x1)
(= arg1_2 bind_x3)
(= r2 within)
(= n2 no)
(= arg2_1 bind_x2)
(= arg2_2 bind_x3)
(= r3 neq)
(= n3 no)
(= arg3_1 bind_x2)
(= arg3_2 bind_x1)
)))(check-sat)
(get-model)