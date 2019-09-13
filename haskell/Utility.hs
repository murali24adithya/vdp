module Utility where

import Grammar

import Data.Dynamic
import Data.Maybe
import qualified Data.Map.Strict as Map


-- Appends an element to a list without duplication
-- TODO: consider using sets instead of lists and converting back and forth, or switching to sets
nodupappend :: Eq a => a -> [a] -> [a]
nodupappend e lst = if elem e lst then lst else lst ++ [e]

-- Appends two lists without duplication
-- TODO: consider using sets instead of lists and converting back and forth, or switching to sets
lstunion :: Eq a => [a] -> [a] -> [a]
lstunion = \lst1 lst2 -> foldr nodupappend lst1 lst2

----------------------------------------------------------------------------

-- Function that plays the same role as 'split' in Python.
-- Mainly used for String formatting in IO.
split :: String -> Char -> [String]
split "" _ = []
split xs c = let (ys, zs) = break (== c) xs
             in  if null zs then [ys] else ys : split (tail zs) c
----------------------------------------------------------------------------

-- Utility functions to collect variables from a given formula/term
collect_vars_labelterm :: LabelTerm -> [String]
collect_vars_labelterm (Labelof a) = [a]
collect_vars_labelterm (Obj a) = []

collect_vars_intterm :: IntTerm -> [String]
collect_vars_intterm Zero = []
collect_vars_intterm (Suc a) = collect_vars_intterm a
collect_vars_intterm (Count a) = collect_vars_labelterm a

collect_vars :: Formula -> [String]
collect_vars (LabRel a b c) = lstunion (collect_vars_labelterm b) (collect_vars_labelterm c)
collect_vars (IntRel a b c) = lstunion (collect_vars_intterm b) (collect_vars_intterm c)
collect_vars (And a b) = lstunion (collect_vars a) (collect_vars b)
collect_vars (Or a b) = lstunion (collect_vars a) (collect_vars b)
collect_vars (Implies a b) = lstunion (collect_vars a) (collect_vars b)
collect_vars (Not a) = collect_vars a
collect_vars (A var formula) = filter (\e -> e /= var) (collect_vars formula)
collect_vars (E var formula) = filter (\e -> e /= var) (collect_vars formula)
collect vars _ = []

----------------------------------------------------------------------------

-- Utility functions on datatype of models

-- Look up the value of a relation (true/false) or a function in a model
-- Input given is the name of a relation and the input argument(s)
model_lookup :: String -> String -> Model -> String
model_lookup rel key mo = fromJust (Map.lookup key (fromJust (Map.lookup rel mo)))

{-
model_lookup :: String -> Dynamic -> [(String, Dynamic)] -> Dynamic
model_lookup rel key lt =
      let functable = (fromDynamic (fromJust (lookup rel lt))) in
      let keyval = (fromDynamic key) in
        if ((functable /= Nothing) && (keyval /= Nothing))
              then (toDyn (fromJust (lookup (fromJust keyval) (fromJust functable) )))
              else (toDyn "ERROR")
-}

-- Insert a key and value within a certain relation.
-- Inputs given are name of the relation, the key and the value.
-- Note that the key can be the string representation of a list.
-- The insert OVERWRITES any existing value for the key with the given one.
model_insert :: String -> String -> String -> Model -> Model
model_insert rel key val mo =
            let lookup_table_update = Map.insert key val (fromJust (Map.lookup rel mo)) in
            Map.insert rel lookup_table_update mo

------------------------------------------------------------

-- Functions to extract elements out of triples.
-- Mainly used to convert premodels (gotten from 'execute' in IOinterface) to models.
triple_ff :: (a,b,c) -> a
triple_ff (x,y,z) = x

triple_ss :: (a,b,c) -> b
triple_ss (x,y,z) = y

triple_tt :: (a,b,c) -> c
triple_tt (x,y,z) = z



