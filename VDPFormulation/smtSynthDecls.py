from vdputils import *

import itertools


def genGuardDeclsSmtSynth(problem_dict):
  num_quants = problem_dict['num-quants']
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(label_datatype)

  one_of_labels = [['=','dummy-label',label] for label in label_set]
  label_value_constraint_formula = ['or'] + one_of_labels
  label_value_constraint_prettyformula = prettyFormula(label_value_constraint_formula)

  res = ''
  for i in range(num_quants):
    label_for_guard_i = 'lbl_g{0}'.format(str(i+1))
    res = res + '(declare-const {0} {1})\n'.format(label_for_guard_i,label_datatype)
    res = res + '(assert {0})\n'.format(label_value_constraint_prettyformula).replace('dummy-label',label_for_guard_i)
    res = res + '(define-fun g{0} ((x {1})) Bool\n(labelOf x {2})\n)\n\n'.format(str(i+1),obj_datatype,label_for_guard_i)
  
  return res


def genFormulaDeclsSmtSynth(problem_dict):
  #Currently only implementing conjunctive formulae with arbitrary number of conjunctions
  num_quants = problem_dict['num-quants']
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(label_datatype)

  #A flag for every atom, and the satisfying assignment will say wheich atoms are used
  #An atom is a relation with a specific tuple of arguments
  var_list = ['x'+str(i+1) for i in range(num_quants)]
  #Atoms corresponding to equality/disequalty between variables
  product_list_1 = [['eq','neq'],var_list,var_list]
  list_of_atoms_1 = [list(x) for x in itertools.product(*product_list_1)]
  #Atoms corresponding to labelOf relation
  product_list_2 = [['labelOf'],var_list,list(label_set)]
  list_of_atoms_2 = [list(x) for x in itertools.product(*product_list_2)]
  #Atoms corresponding to other relations
  remaining_rels = ['leftOf','rightOf','above','below','within']
  product_list_3 = [remaining_rels] + ([var_list]*2)
  list_of_atoms_3 = [list(x) for x in itertools.product(*product_list_3)]
  
  list_of_atoms = list_of_atoms_1 + list_of_atoms_2 + list_of_atoms_3

  #Functions to make a flag and an evaluation term out of list representing an atom
  mkFlag = lambda x: '_'.join(x)
  ##mkTerm = lambda x: '(' + ' '.join(x) + ')'

  res = ''
  #print(list_of_atoms)
  #Declaring flags for each atom
  for atom in list_of_atoms:
    res = res + '(declare-const {0} Bool)\n'.format(mkFlag(atom))
  res = res + '\n'

  #Defining the formula evaluation using the flags
  formula_evaluation_conjuncts = [['=>',mkFlag(atom),atom] for atom in list_of_atoms]
  formula_evaluation = ['and'] + formula_evaluation_conjuncts

  var_decl = ['(x' + str(i+1) + ' {0})'.format(obj_datatype) for i in range(num_quants)]
  
  res = res + '(define-fun f ({0}) Bool\n{1}\n)\n'.format(' '.join(var_decl),prettyFormula(formula_evaluation))

  return res


def genLabelDecls(problem_dict):
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(label_datatype)
  
  res = ''
  for label in label_set:
    label_indicator = 'constval_' + label
    res = res + '(declare-const {0} {1})\n'.format(label_indicator,label_datatype)
    res = res + '(assert (= {0} {1}))\n'.format(label_indicator,label)
  
  return res



def genSmtSynthDecls(problem_dict):
  num_quants = problem_dict['num-quants']
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  
  res = ''
  
  #Quantifier declarations
  for i in range(num_quants):
    res = res + '(declare-const q{0} Bool)\n'.format(str(i+1))  
  res = res + '\n'
  
  #Guard declarations
  res = res + genGuardDeclsSmtSynth(problem_dict)
  res = res + '\n'

  #Flags to indicate what labels correspond to what bitvectors
  ##Not necessary for the synthesis
  ##But helps obtain labels after solver comes back
  res = res + genLabelDecls(problem_dict)
  res = res + '\n'

  #Inner formula declaration
  res = res + genFormulaDeclsSmtSynth(problem_dict)
  res = res + '\n'

  #Candidate flag declaration
  num_candidates = len(problem_dict['candidateimgs'])
  for i in range(num_candidates):
    res = res + '(declare-const c{0} Bool)\n'.format(str(i+1))
  res = res + '\n'

  res = res + '\n'

  return res


def writeSmtSynthDecls(problem_dict,filename,write_or_append):
  decls = genSmtSynthDecls(problem_dict)
  filehandle = open(filename,write_or_append)
  filehandle.write(decls)
  filehandle.close()