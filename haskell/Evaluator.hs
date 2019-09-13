module Evaluator where

import Grammar
import Utility
import Synthesis

import Data.Dynamic
import Data.Maybe
import qualified Data.Map.Strict as Map


{-
This file describes the semantics of formulae by evaluating them with respect to a model.
The model is stored as a list of pairs, the first component being the name of the relation, and the
second component a list of all tuples that the relation makes true. Functions are stored similarly,
with inputs and outputs being the tuples.
However, for relations and functions such as equality, integer comparison, etc., the functions are
implemented directly in the semantics.
-}

--Lower level function evaluate LabelTerm terms
evaluate_labelterm :: LabelTerm -> Model -> Assgn -> LabelTerm
evaluate_labelterm (Obj a) _ _ = Obj a
evaluate_labelterm (Labelof a) mo assgn = (read (model_lookup "label" (fromJust (lookup a assgn)) mo) :: LabelTerm )

-- Lower level function to evaluate IntTerm terms
evaluate_intterm :: IntTerm -> Model -> Assgn -> Int
evaluate_intterm Zero _ _ = 0
evaluate_intterm (Suc it) mo assgn = (evaluate_intterm it mo assgn) + 1
evaluate_intterm (Count lt) mo assgn = (read (model_lookup "count" (show (evaluate_labelterm lt mo assgn)) mo) :: Int )


-- Evaluate formulae given a model
evaluate :: Formula -> Model -> Assgn -> Bool
evaluate formula model assignment = evaluate_aux formula model assignment
  where
  evaluate_aux Top _ _ = True
  evaluate_aux Bot _ _ = False
  evaluate_aux (LabRel "EqLabel" lt1 lt2) mo assgn = (evaluate_labelterm lt1 mo assgn) == (evaluate_labelterm lt2 mo assgn)
  evaluate_aux (IntRel "EqInt" it1 it2) mo assgn = (evaluate_intterm it1 mo assgn) == (evaluate_intterm it2 mo assgn)
  evaluate_aux (IntRel "LessInt" it1 it2) mo assgn = (evaluate_intterm it1 mo assgn) < (evaluate_intterm it2 mo assgn)
  evaluate_aux (And f1 f2) mo assgn = (evaluate_aux f1 mo assgn) && (evaluate_aux f2 mo assgn)
  evaluate_aux (Or f1 f2) mo assgn = (evaluate_aux f1 mo assgn) || (evaluate_aux f2 mo assgn)
  evaluate_aux (Implies f1 f2) mo assgn = (not (evaluate_aux f1 mo assgn)) || (evaluate_aux f2 mo assgn)
  evaluate_aux (Not f) mo assgn = not (evaluate f mo assgn)
  evaluate_aux (A var f) mo assgn = foldr (&&) True [evaluate_aux f mo (assgn ++ [(var,object)]) | object <- objects]
  evaluate_aux (E var f) mo assgn = foldr (||) False [evaluate_aux f mo (assgn ++ [(var,object)]) | object <- objects]
  objects = (Map.keys (fromJust (Map.lookup "label" model)))


{-
evaluate_labelterm :: LabelTerm -> [(String,Dynamic)] -> [(String, String)] -> LabelTerm
evaluate_labelterm (Obj a) _ _ = Obj a
evaluate_labelterm (Labelof a) mo assgn = fromJust (fromDynamic (model_lookup "label" (toDyn (fromJust (lookup a as))) mo))

evaluate_intterm :: IntTerm -> [(String, Dynamic)] -> [(String,String)] -> Int
evaluate_intterm Zero _ _ = 0
evaluate_intterm (Suc it) mo assgn = (evaluate_intterm it mo assgn) + 1
evaluate_intterm (Count lt) mo assgn = fromJust (fromDynamic (model_lookup "count" (toDyn (evaluate_labelterm lt mo assgn)) mo))


evaluate :: Formula -> [(String,Dynamic)] -> [(String, String)] -> Bool
evaluate Top _ _ = True
evaluate Bot _ _ = False
evaluate (LabRel "EqLabel" lt1 lt2) mo assgn = (evaluate_labelterm lt1 mo assgn) == (evaluate_labelterm lt2 mo assgn)
evaluate (IntRel "EqInt" it1 it2) mo assgn = (evaluate_intterm it1 mo assgn) == (evaluate_intterm it2 mo assgn)
evaluate (IntRel "LessInt" it1 it2) mo assgn = (evaluate_intterm it1 mo assgn) < (evaluate_intterm it2 mo assgn)
evaluate (And f1 f2) mo assgn = (evaluate f1 mo assgn) && (evaluate f2 mo assgn)
evaluate (Or f1 f2) mo assgn = (evaluate f1 mo assgn) || (evaluate f2 mo assgn)
evaluate (Implies f1 f2) mo assgn = (not (evaluate f1 mo assgn)) || (evaluate f2 mo assgn)
evaluate (Not f) mo assgn = not (evaluate f mo assgn)
-}

{-
-------------------------------------------------------------
count_model = [("count",Map.fromList [(show (Obj "l1"),"3"),(show(Obj "l2"), "5")])]
label_model = [("label",Map.fromList [("_x1",show (Obj "l1")),("_x2",show (Obj "l2"))])]
model = Map.fromList (count_model ++ label_model)
assgn = [("x1","_x1"),("x2","_x2"),("x3","_x1")]

test = \e -> do
              print (evaluate_intterm (Count (Labelof "x2")) model assgn)
              print (evaluate_intterm (Suc (Count (Labelof "x2"))) model assgn)
              print (evaluate_intterm Zero model assgn)
              print (evaluate (LabRel "EqLabel" (Labelof "x1") (Labelof "x3")) model assgn)
              print (evaluate (LabRel "EqLabel" (Labelof "x1") (Labelof "x2")) model assgn)
              print (evaluate (IntRel "LessInt" (Count (Labelof "x1")) (Count (Labelof "x3"))) model assgn)

--g inp ls = fromJust (lookup inp ls)
--f key inp mo = g inp (fromJust (fromDynamic (fromJust (lookup key mo))))
--a = [("hello" :: String,(toDyn [(2 :: Int,"3")])),("hi",(toDyn [("3",2)]))]
--b = (f ("hello" :: String) (2 :: Int) (a :: [(String,Dynamic)])) :: String

-------------------------------------------------------------------------
-}