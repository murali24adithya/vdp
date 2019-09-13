module Classifier where

import Grammar
import Utility
import Synthesis
import Evaluator

import Data.Dynamic
import Data.Maybe
import Data.Set (Set)
import qualified Data.Set as Set
import qualified Data.Map.Strict as Map


-- Find common formulae satisfied by a list of models
find_pattern :: [Model] -> Int -> Int -> [Formula]
find_pattern models cost num_vars =
    let label_model = \model -> (fromJust (Map.lookup "label" model)) in
    let labelslist = \model ->  [label | Obj label <- [(read prelabel) :: LabelTerm | prelabel <- Map.elems (label_model model)]] in
    let all_labels = (foldr (\e lst -> lstunion lst (labelslist e)) [] models) in
    let formulas = gen_formula cost ["x" ++ show i | i <- [1..num_vars]] all_labels in
    let sat_formulas = [filter (\formula -> evaluate formula model []) formulas | model <- models] in
    let formulas = foldr (\e lst -> [Set.intersection e (lst !! 0)] ) [(Set.fromList (sat_formulas !! 0))] (map Set.fromList sat_formulas) in
    Set.toList (formulas !! 0)


------------------------------------------------------------------------------

-- Make models uniform in universe of objects/class labels, etc.
-- This is to ensure that classifiers based on a vocabulary specific to one model in terms of class labels, etc (not object names) can be evaluated on another model.
makeuniform :: [Model] -> [Model]
makeuniform models =
    let labelslist = \model -> Map.elems (fromJust (Map.lookup "label" model)) in
    let all_labels = (foldr (\e lst -> lstunion lst (labelslist e)) [] models) in
    let defaultcounts = Map.fromList [(label,"0") | label <- all_labels] in
    let count_update = \count_model -> Map.unionWith (\x y -> y) defaultcounts count_model in
    let model_update = \model -> Map.insert "count" (count_update (fromJust (Map.lookup "count" model))) model in
    map model_update models
----------------------------------------------------------------------------

-- Find formulae satisfied by exactly one model
-- TODO: must also return which model (in terms of index, say)
find_classifier :: [Formula] -> [Model] -> [Formula]
find_classifier formulas testmodels =
     let getmodels = \formula -> filter (\model -> evaluate formula model []) testmodels in
     filter (\formula -> length (getmodels formula) == 1 ) formulas

-- End-to-End finding classifier formulae below a certain cost and on a certain number of variables.
-- Function must also be given training models and test models.
solve :: Int -> Int -> [Model] -> [Model] -> [Formula]
solve cost num_vars trainmodels testmodels = find_classifier (find_pattern trainmodels cost num_vars) testmodels

---------------------------------------------------------------------------
