module Synthesis where

import Grammar
import Utility

import Data.Dynamic
import Data.Maybe
import qualified Data.Map.Strict as Map
import Data.Array as Array

{-
- TODO: The implementation below may be made more efficient(?) by precomputing
pairs of formulas for conjunctive, disjunctive, and implication
forms. There's also repetition in computing negation forms (and other
similar places such as base predicates), as the
only formulas whose negation will be in k but not in k-1 are those
of the formulas in k-1 but not in k-2.

- Dynamic programming for relation symbols and terms has also been implemented, but not very efficiently.
TODO: They should ideally be within gen_qf_formula itself so that they're not computed for each step of the dp
implementation at the formula level.

- Similar to formulae, terms also repeat greatly between successive cost values.
TODO: Must look into more
efficient implementation.

- TODO: Must modify dp procedure to accomodate functionally specified costs for terms/formulae.
--- Lower costs expressed as smaller numbers versus negative costs?
-}

{-
Cost model:
- Formulae are evaluated for cost only on the quantifier-free part.
- True and False are worth nothing.
- Boolean operators are all worth 1.
- Constants are worth 1.
- Relation symbols themselves are worth 1.
- Function symbols themselves are worth 1.
- Each variable of LabelTerm type is worth 1.
- Quantifiers have no cost at all.
- Total cost is additive in terms of written expression upon these above rules, for the most part.

The functions below are meant to capture terms/formulae below a certain cost
But a list of allowed variables that can appear in the formula and label constants are also required.
-}


-- Label terms: constant cost and easily determined. Therefore not defined explicitly here.
--gen_labterms :: Int -> [LabelTerm]

-- Label relations below a given cost
gen_labrels :: Int -> [String] -> [String] -> [Formula]
gen_labrels k varslist labelslist = if k < 3 then [] else [LabRel rel lt1 lt2 | rel <- labrels, lt1 <- labelterms_all, lt2 <- labelterms_nonconst]
                        where labelterms_all = [Obj label | label <- labelslist] ++ [Labelof var | var <- varslist]
                              labelterms_nonconst = [Labelof var | var <- varslist]


-- Returns an array of entries of integer terms below a cost value (from costs of 1 until input value)
-- The Boolean parameter controls whether pure constants can be included. This is to avoid relating pure constants as a generated formula.
gen_intterms :: Int -> [String] -> [String] -> Bool -> Array Int [IntTerm]
gen_intterms measure varslist labelslist flag = dparray
    where
    dpcomp 1 flag = if flag then [Zero] else []
    dpcomp 2 flag = [Count label | label <- labelterms] ++ (if flag then [Zero, Suc Zero] else [])
    dpcomp k flag = [Suc intterm | intterm <- (dparray ! (k-1))]
    labelterms = [Obj label | label <- labelslist] ++ [Labelof var | var <- varslist]
    valarray = Array.listArray (1, measure) [dpcomp cost flag | cost <- [1..measure]]
    dparray = Array.listArray (1, measure) [foldr (\i lst -> lstunion lst (valarray ! i)) [] [1..cost] | cost <- [1..measure]]

-- Integer relations below a given cost. Relation symbols have constant cost 1.
gen_intrels :: Int -> [String] -> [String] -> [Formula]
gen_intrels k varslist labelslist = if k < 3 then [] else
                          let foldfunc = (\i lst -> lstunion lst [IntRel rel it1 it2 | rel <- intrels, it1 <- (inttermslist_all ! i), it2 <- (inttermlist_nonconst ! (k-1-i))]) in
                            foldr foldfunc [] [1..(k-2)]
                                where
                                inttermslist_all = (gen_intterms (k-2) varslist labelslist True)
                                inttermlist_nonconst = (gen_intterms (k-2) varslist labelslist False)


-- All quantifier-free formulae below a certain cost
gen_qf_formula :: Int -> [String] -> [String] -> [Formula]
gen_qf_formula measure varslist labelslist = dparray ! measure
    where
    lazycomp 0 = [] --[Top,Bot]
    lazycomp k = let negforms = [Not f | f <- dparray ! (k -1)] in
             let conjforms = foldr (\i lst -> lstunion lst [And f1 f2 | f1 <- (dparray ! i), f2 <- (dparray ! (k-1-i))]) [] [0..(k-2)] in
             let disjforms = foldr (\i lst -> lstunion lst [Or f1 f2 | f1 <- (dparray ! i), f2 <- (dparray ! (k-1-i))]) [] [0..(k-2)] in
             let implforms = foldr (\i lst -> lstunion lst [Implies f1 f2 | f1 <- (dparray ! i), f2 <- (dparray ! (k-1-i))]) [] [0..(k-2)] in
             let labrelforms = gen_labrels k varslist labelslist in
             let intrelforms = gen_intrels k varslist labelslist in
               negforms ++ conjforms ++ disjforms ++ implforms ++ labrelforms ++ intrelforms
    valarray = Array.listArray (0,measure) [lazycomp cost | cost <- [0..measure] ]
    dparray = Array.listArray (0,measure) [(foldr (\e lst -> lstunion lst (valarray ! e)) [] [0..cost]) | cost <- [0..measure] ]

------------------------------------------------------------------------------------------------------

{- We can now synthesise quantified formulas of a certain number of variables and a certain cost by
using the above procedure for quantifier-free formulas and choosing to put various orders of quantifiers
on the variables (whose names we know because we are given them).
-}

-- Adds quantifiers to a formula depending on its free variables (all possible combinations)
quantify_formula :: Formula -> [Formula]
quantify_formula f = quantify_formula_aux f (collect_vars f)
  where
  quantify_formula_aux formula [] = [formula]
  quantify_formula_aux formula vars =
    let  univforms = [A (vars !! 0) ff | ff <- (quantify_formula_aux formula (drop 1 vars)) ] in
    let existforms = [E (vars !! 0) ff | ff <- (quantify_formula_aux formula (drop 1 vars)) ] in
    univforms ++ existforms

--formula_example = A "x" (IntRel "EqInt" (Count (Labelof "x")) (Count (Labelof "y")))


{- Synthesise sentences below a certain cost over a given list of variable
names and labels -}
gen_formula :: Int -> [String] -> [String] -> [Formula]
gen_formula measure varslist labelslist =
    let qflist = gen_qf_formula measure varslist labelslist in
    foldr (++) [] (map quantify_formula qflist)

----------------------------------------------------------------------------------------------------

