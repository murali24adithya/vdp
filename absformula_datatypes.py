from objs_and_imgs import *

# Defines the type of variable indicating which quantifier will be used
def declare_datatype_quantifier():
  result = "(declare-datatypes () ((Quantifier all one)))"
  preamble =";Datatype for indicating which quantifier to use in formula\n"
  return preamble + result

# Defines a datatype 'binding' that indicates which arguments (for relations occuring in the formula) bind to which variables/constants
# Constants are extraneous objects, so they get bindings as well.
# The standard prefix is 'bind_' to indicate what the binding indicates
def declare_datatype_binding(d,num_vars):
  temp = ""
  for i in range(1,num_vars+1):
    temp = temp + " bind_x" + str(i)
  
  temp = temp + "\n;bindings for extraneous objects\n"
  exobs = extraneous_objs(d)
  for exob in exobs:
    temp = temp + " bind_" + exob

  result = "(declare-datatypes () ((Binding\n" + temp + "\n)))"

  preamble = ";Datatype of indirections to indicate what the arguments of relations in the abstract baseformula are-- i.e, a datatype of bindings \n" + ";Constants/extraneous objects have bindings as well\n"
  return preamble + result

def define_fun_bindlook(d,num_vars):
  args = ""
  for i in range(1,num_vars+1):
    args = args + "(x" + str(i) + " Obj)"
  
  temp = " nullobj" 
  exobs = extraneous_objs(d)
  for exob in exobs:
    temp = "(ite (= bind " + ("bind_" + exob) + ") " + (exob) + "\n" + temp + ")"

  temp = ";Bindings for extraneous objects\n" + temp
  
  for i in range(1,num_vars+1):
    temp = "(ite (= bind " + ("bind_x" + str(i)) + ") " + ("x" + str(i)) + "\n" + temp + ")"
  
  result = "(define-fun bindlook (" + args + "(bind Binding)) Obj\n" + temp + "\n)"

  preamble = ";Defines how bindings correspond to valuations\n" + ";Default lookup is nullobj, but it is used since the function is defined for all bindings\n"
  return preamble + result

def define_operator_meanings():
  result = "(declare-datatypes () ((Ops opand opor opimpl)))\n" + \
  "(declare-datatypes () ((IfNeg yes no)))\n" + \
  "(define-fun opeval ((op Ops) (arg1 Bool) (arg2 Bool)) Bool\n" + \
  " (ite (= op opand) (and arg1 arg2)\n" + \
  " (ite (= op opor) (or arg1 arg2)\n" + \
  "  (=> arg1 arg2)\n" + \
  ")))\n" + \
  "(define-fun negeval ((val IfNeg)(arg Bool)) Bool\n" + \
  " (ite (= val yes)\n" + \
  "  (not arg)\n" + \
  "  arg\n" + \
  "))"

  preamble = ";Defines symbols used to indicate which boolean operators will be used, including negations, and their meanings as well\n"
  return preamble + result


def define_quantifier_vars(num_vars):
  result = ""
  for i in range(1,num_vars+1):
    result = result + "(declare-const q" + str(i) + " Quantifier)\n"
  return result

def define_operator_vars(num_rels):
  result = ""
  for i in range(1,num_rels):
    result = result + "(declare-const op" + str(i) + " Ops)\n"
  return result

def define_negation_vars(num_rels):
  result = ""
  for i in range(1,num_rels+1):
    result = result + "(declare-const n" + str(i) + " IfNeg)\n"
  return result

def define_relation_vars(num_rels):
  result = ""
  for i in range(1,num_rels+1):
    result = result + "(declare-const r" + str(i) + " Rels)\n"
  return result

def define_relarg_vars(num_rels,arity):
  result = ""
  for i in range(1,num_rels+1):
    for j in range(1,arity+1):
      result = result + "(declare-const arg" + str(i) + "_" + str(j) + " Binding)\n"
  return result


def define_quantifier_constraints(existential,num_vars):
  result = ""
  if existential == 'yes':
    for i in range(1,num_vars+1):
      result = result + "(assert (= q" + str(i) + " one))" + "\n"
    result = result + "\n"
  return result

def define_operator_constraints(conjuncts,num_rels):
  result = ""
  if conjuncts == 'yes':
    for i in range(1,num_rels):
      result = result + "(assert (= op" + str(i) + " opand))" + "\n"
    result = result + "\n"
  return result

def define_labelusage_constraints(labelconstraint,num_rels):
  result = ""
  if labelconstraint == 'off':
    for i in range(1,num_rels+1):
      result = result + "(assert (not (= r" + str(i) + " labelOf)))\n"
    result = result + "\n"
  elif labelconstraint == 'noneg':
    for i in range(1,num_rels+1):
      result = result + "(assert (=> (= r" + str(i) + " labelOf) (= n" + str(i) + " no)))\n"
    result = result + "\n"
  return result