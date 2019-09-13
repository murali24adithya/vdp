module Main where

import Grammar
import Utility
import Synthesis
import Evaluator
import Classifier
import IOinterface

import Trainmodels
import Testmodels

import Data.Dynamic
import Data.Maybe
import Data.List as List
import Data.Set (Set)
import qualified Data.Set as Set
import Data.Ord
import qualified Data.Map.Strict as Map


-- Examples of models to search common satisfiable formulae from
count_model_1 = [("count",Map.fromList [(show (Obj "l1"),"3"),(show(Obj "l2"), "5"), (show (Obj "l3") ,"2")])]
label_model_1 = [("label",Map.fromList [("_x1",show (Obj "l1")),("_x2",show (Obj "l2"))])]
model_1 = Map.fromList (count_model_1 ++ label_model_1)

count_model_2 = [("count",Map.fromList [(show (Obj "l1"),"2"),(show(Obj "l2"), "4"), (show (Obj "l3") ,"1")])]
label_model_2 = [("label",Map.fromList [("_x1",show (Obj "l1")),("_x2",show (Obj "l2"))])]
model_2 = Map.fromList (count_model_2 ++ label_model_2)

-------------------------------------------------------------------------

num_vars = 0
cost = 5

trainmodels = map makemodel train_models
testmodels = map makemodel test_models
uniformmodels = makeuniform (trainmodels ++ testmodels)
uniform_trainmodels = take (length trainmodels) uniformmodels
uniform_testmodels = drop (length trainmodels) uniformmodels

solution = solve cost num_vars uniform_trainmodels uniform_testmodels

main = putStrLn (foldr (\formula str-> str ++ (show formula)++ "\n") "\n" solution)






