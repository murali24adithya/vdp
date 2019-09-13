module IOinterface where


import Grammar
import Utility
import Synthesis
import Evaluator
import Classifier

import System.IO
import Data.List as List
import Data.Ord
import qualified Data.Map.Strict as Map


-- Format input string from YOLO into a list of premodels (not parsed)
-- The return type is a 'premodel'
execute :: String -> [[(String,String,[String])]]
execute str =
    let val1 = drop 1 (take ((length str) - 1) (split str 'E')) in
    let val2 = (map (\e -> drop 1 (split e '\n')) val1) in
    let val3 = take ((length val2) - 1) val2 in
    let labelof = \x -> (split x ':') !! 0 in
    let all_labels = \lst -> [labelof (lst !! i) | i <- [0,2..((length lst)-1)]] in
    let confidence = \x -> ((split x ':') !! 1) in
    let all_confidence = \lst -> [confidence (lst !! i) | i <- [0,2..((length lst)-1)]] in
    let bounding_box = \x -> map (\e -> (split e '=') !! 1) (split ( (split x ':') !! 1) ',') in
    let all_bounding = \lst -> [bounding_box (lst !! i) | i <- [1,3..((length lst)-1)]] in
    let get_all_obj = \lst -> zip3 (all_labels lst) (all_confidence lst) (all_bounding lst) in
    map get_all_obj val3


-- Make model from premodel (which is what 'execute' in IOinterface would return, modulo the IO monad part)
-- TODO: Consider moving to utility and make a premodel type
makemodel :: [(String,String,[String])] -> Model
makemodel rawmodel =
    let num_elems = length rawmodel in
    let label_map = [("_" ++ show i,show (Obj (triple_ff (rawmodel !! i)))) | i <- [0..(num_elems-1)]] in
    let count_map = map (\x -> ((snd (x !!0)),show (length x))) (List.groupBy (\x y -> (snd x) == (snd y)) (sortBy (comparing snd) label_map)) in
    Map.fromList ([("label",Map.fromList label_map)] ++ [("count",Map.fromList count_map)])



-- Interface through which the whole system must be channeled.
-- File read/write and output of formula to be done here.
input_text = do {
                handle <- openFile "test.txt" ReadMode;
                content <- hGetContents handle;
                let text = execute content in
                return text
                }

-------------------------------------------------------------------------
raw_model = [[("cat"," 100%",["360","45","820","484"])],
 [("bed"," 63%",["11","284","613","494"]),
  ("dog"," 100%",["23","129","285","447"]),
  ("cat"," 100%",["280","92","600","411"])],
 [("dog"," 100%",["0","59","94","199"]),
  ("dog"," 99%",["163","12","237","203"]),
  ("dog"," 90%",["280","66","336","208"]),
  ("cat"," 100%",["338","133","399","203"]),
  ("cat"," 100%",["97","103","163","190"]),
  ("cat"," 98%",["226","89","283","192"])],
 [("dog"," 98%",["151","30","210","132"]),
  ("dog"," 97%",["34","27","110","143"]),
  ("cat"," 91%",["95","50","165","160"]),
  ("cat"," 99%",["212","35","263","120"])],
 [("dog"," 56%",["238","176","567","556"]),
  ("cat"," 97%",["16","191","268","545"]),
  ("cat"," 66%",["239","47","430","245"])]]

--------------------------------------------------------------------------
