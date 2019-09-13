from parsetree import *
from absformula_datatypes import *
from absformula_gen import *
from rels import *
from objs_and_imgs import *
from io import *
from utilities import *
from alternaterels import *
#from smtoutput import *

import sys


# Dictionary has to store the following key-value pairs:
# elem -> points to a dictionary of type image id -> [listof object ids belonging to that image] 
# label -> points to a dictionary of type obj id -> label
# top_border -> points to a dictionary of type obj id -> top coordinate
# similarly for bottom_border, left_border and right_border
# obj id, img id, labels are all strings
# borders are numbers


def assert_formula_on_train(train_imgs):
  result = ""
  for img in train_imgs:
    result = result + "(assert (formula_level_1 " + img + "))\n"
  return result

def solve_standard(test_imgs):
  result = ""

  for posimg in test_imgs:
    result = result + "(push)\n"
    result = result + "(echo \"Results for " + posimg + "\")\n"
    result = result + "(assert (formula_level_1 " + posimg + "))\n"
    for  negimg in test_imgs:
      if negimg != posimg:
        result = result + "(assert (not (formula_level_1 " + negimg + ")))\n"
    result = result + "(check-sat)\n" + "(get-model)\n"
    result = result + "(pop)\n\n"

  return result


# arg = {'elem':{'t1':['t1_1','t1_2'],'t2':['t2_1','t2_2']}, 'label': {'t1_1':'cat','t1_2':'sofa'}, 'top_border':{'t1_1': 0,'t1_2':0},'bottom_border':{'t1_1': 0,'t1_2':0},'left_border':{'t1_1': 0,'t1_2':0},'right_border':{'t1_1': 0,'t1_2':0}}


# num_vars indicating number of variables quantified in the formula
# num_rels indicating the number of relation instances in the formula
# arity denoting the uniform arity of all relations
# Given num_vars = k, formula will contain the variables x1, x2... xk
# It will also contain the quantifier variables b1, b2... bk
# Given num_rels = k and arity = m, abstract formula will contain the relation type variables r1, r2... rk and each rj will contain the arguments argj_1, argj_2... argj_m


def smt_file_gen(problem_dict,smtdict,config_dict,solve_str='standard'):
  num_vars = config_dict['num_vars']
  num_rels = config_dict['num_rels']
  arity = config_dict['arity']
  existential = config_dict['existential']
  conjuncts = config_dict['conjuncts']
  labelconstraint = config_dict['labelconstraint']
  counting = config_dict['counting']

  result = ""
  result = result + declare_datatype_Img(smtdict) + "\n"
  result = result + declare_datatype_Obj(smtdict) + "\n"
  result = result + define_fun_inImg(smtdict) + "\n"
  #result = result + define_fun_label(smtdict) + "\n"
  result = result + declare_datatype_rels(counting) + "\n"
  result = result + define_fun_precompute_re(problem_dict,counting) + "\n"
  result = result + declare_datatype_quantifier() + "\n"
  result = result + declare_datatype_binding(smtdict, num_vars) + "\n"
  result = result + define_fun_bindlook(smtdict, num_vars) + "\n"
  result = result + define_operator_meanings() + "\n"
  result = result + define_quantifier_vars(num_vars) + "\n"
  result = result + define_operator_vars(num_rels) + "\n"
  result = result + define_negation_vars(num_rels) + "\n"
  result = result + define_relation_vars(num_rels) + "\n"
  result = result + define_relarg_vars(num_rels, arity) + "\n"

  result = result + define_quantifier_constraints(existential, num_vars)
  result = result + define_operator_constraints(conjuncts, num_rels)
  result = result + define_labelusage_constraints(labelconstraint, num_rels)

  result = result + define_baseformula(num_rels, num_vars, arity,counting) + "\n"
  result = result + define_formula_levels(num_vars,counting) + "\n"
  result = result + "\n" + assert_formula_on_train(problem_dict['train'].keys()) + "\n"

  if solve_str == 'standard':
    result = result + solve_standard(problem_dict['test'].keys())
  else:
    result = result + solve_str + "(check-sat)\n" + "(get-model)"

  return result
