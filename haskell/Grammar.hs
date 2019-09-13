module Grammar where

import Data.Dynamic
import Data.Maybe
import qualified Data.Map.Strict as Map

{-
Formulas and terms must be derived from Eq and Ord to do comparison based sorting
and to remove duplicates. They must also be derived from Read and Show because
we will later use (in the construction of a first-order model) a String representation
as a top type for the type of any formula/term.
-}

data LabelTerm =  Labelof String -- label of var
                | Obj String   -- object label
                deriving (Read, Show, Eq, Ord)

-- must add support for more general function application over terms
data IntTerm =  Zero -- integer base constant
              | Suc IntTerm -- successor function
              | Count LabelTerm --count of number of elements with a given label
              deriving (Read, Show, Eq, Ord)

data Formula =  Top
              | Bot
              | LabRel String LabelTerm LabelTerm
              | IntRel String IntTerm IntTerm
              | And Formula Formula
              | Or Formula Formula
              | Implies Formula Formula
              | Not Formula
              | A String Formula -- universal quantifier
              | E String Formula -- existential quantifier
              deriving (Read, Show, Eq, Ord)

labrels = ["EqLabel"]
intrels = ["EqInt", "LessInt"]

---------------------------------------------------------------------------

{- The datatype of models is a list of lookup tables indexed by the name
of the relation/function. Some standard relations such as 'label' will determine
the total set of objects and class labels present in the model (set of all keys
and values, respectively). All data is stored as a string. This is why it is essential
that all elements of the grammar are 'Showable' and 'Readable'.

Therefore, an outer map indexes relation names to lookup tables, where each lookup
table indexes the input values (written as a list) to the output value.
For relations, the output value is the string representation of True or False.
-}

type Model = Map.Map String (Map.Map String String)--[(String, [(String,String)])]


{- The type of assignments is similar to that of models. It is a single lookup table,
 and is currently implemented as an associative list. To be replaced with Map
 -}

type Assgn = [(String, String)]