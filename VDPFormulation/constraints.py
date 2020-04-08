from vdputils import *

import itertools
import string


def genTrueConstraints(formula,elems,quants,varindex=1):
  if quants == []:
    return formula
  else:
    quant = quants[0]
    rest_quants = quants[1:]
    if quant == 'true':
      #Universal Quantifier
      #Including vacuity
      univ_formula_base = ['=>',['g'+str(varindex),'x'+str(varindex)],genTrueConstraints(formula,elems,rest_quants,varindex+1)]
      exist_formula_base = ['g'+str(varindex),'x'+str(varindex)]

      univ_formula = [subst(univ_formula_base,'x'+str(varindex),elem) for elem in elems]
      exist_formula = [subst(exist_formula_base, 'x' + str(varindex), elem) for elem in elems]

      univ_formula_final = ['and'] + univ_formula
      exist_formula_final = ['or'] + exist_formula
      return ['and',exist_formula_final,univ_formula_final]
    
    else:
      #Existential Quantifier
      exist_formula_base = ['and',['g'+str(varindex),'x'+str(varindex)],genTrueConstraints(formula,elems,rest_quants,varindex+1)]

      exist_formula = [subst(exist_formula_base,'x'+str(varindex),elem) for elem in elems]

      exist_formula_final = ['or'] + exist_formula
      return exist_formula_final


def genFalseConstraints(formula,elems,quants,varindex=1):
  if quants == []:
    return ['not',formula] #negation pushed all the way in
  else:
    quant = quants[0]
    rest_quants = quants[1:]
    if quant == 'true':
      #Universal Quantifier: acts like existential quantifier
      univ_formula_base = ['and',['g'+str(varindex),'x'+str(varindex)],genFalseConstraints(formula,elems,rest_quants,varindex+1)]
      
      univ_formula = [subst(univ_formula_base,'x'+str(varindex),elem) for elem in elems]

      univ_formula_final = ['or'] + univ_formula
      return univ_formula_final
    
    else:
      #Existential Quantifier: acts like universal quantifier
      exist_formula_base = ['=>',['g'+str(varindex),'x'+str(varindex)],genFalseConstraints(formula,elems,rest_quants,varindex+1)]

      exist_formula = [subst(exist_formula_base,'x'+str(varindex),elem) for elem in elems]

      exist_formula_final = ['and'] + exist_formula
      return exist_formula_final


# def genUnnegatedFalseConstraints(formula,elems,quants,varindex=1):
#   if quants == []:
#     return formula
#   else:
#     quant = quants[0]
#     rest_quants = quants[1:]
#     if quant:
#       #Universal Quantifier
#       univ_formula_base = ['=>',['g'+str(varindex),'x'+str(varindex)],genTrueConstraints(formula,elems,rest_quants,varindex+1)]
      
#       univ_formula = [subst(univ_formula_base,'x'+str(varindex),elem) for elem in elems]

#       univ_formula_final = ['and'] + univ_formula
#       return univ_formula_final
    
#     else:
#       #Existential Quantifier
#       exist_formula_base = ['and',['g'+str(varindex),'x'+str(varindex)],genTrueConstraints(formula,elems,rest_quants,varindex+1)]

#       exist_formula = [subst(exist_formula_base,'x'+str(varindex),elem) for elem in elems]

#       exist_formula_final = ['or'] + exist_formula
#       return exist_formula_final


def genAllQuantsTrueConstraints(formula,elems,num_quants):
  all_quants = [list(quant_choice) for quant_choice in itertools.product(['true','false'],repeat=num_quants)]
  #all_quants = [['true','false']]
  quant_combination_constraint = lambda quant_choice: ['and'] + [['=','q'+str(i+1), 'true' if (quant_choice[i] == 'true') else 'false'] for i in range(len(quant_choice))]
  all_constraints = [['=>',quant_combination_constraint(quant_choice),genTrueConstraints(formula,elems,quant_choice)] for quant_choice in all_quants]
  return ['and'] + all_constraints


def genAllQuantsFalseConstraints(formula,elems,num_quants):
  all_quants = [list(quant_choice) for quant_choice in itertools.product(['true','false'],repeat=num_quants)]
  #all_quants = [['true','false']]
  quant_combination_constraint = lambda quant_choice: ['and'] + [['=','q'+str(i+1), 'true' if (quant_choice[i] == 'true') else 'false'] for i in range(len(quant_choice))]
  all_constraints = [['=>',quant_combination_constraint(quant_choice),genFalseConstraints(formula,elems,quant_choice)] for quant_choice in all_quants]
  return ['and'] + all_constraints


def genOneImgConstraints(formula,num_quants,img_type,img,imgindex=None):
  objs_in_img = [objdict['objid'] for objdict in img]

  if img_type == 't':
    constraints = genAllQuantsTrueConstraints(formula,objs_in_img,num_quants)
    return '(constraint\n' + prettyFormula(constraints) + ')\n\n'

  elif img_type == 'c':
    true_constraints = genAllQuantsTrueConstraints(formula,objs_in_img,num_quants)
    true_constraints = ['=>',['=','c'+str(imgindex),'true'],true_constraints]
    true_constraints_string = '(constraint\n' + prettyFormula(true_constraints) + ')\n\n'

    false_constraints = genAllQuantsFalseConstraints(formula,objs_in_img,num_quants)
    false_constraints = ['=>',['=','c'+str(imgindex),'false'],false_constraints]
    false_constraints_string = '(constraint\n' + prettyFormula(false_constraints) + ')\n\n'

    return true_constraints_string + false_constraints_string

  else:
    raise ValueError('Incorrect image type: must be either \'t\' or \'c\' ')


def genChooseOneCandidateConstraints(num_candidates):
  res = ''
  all_candidates = ['c'+str(i+1) for i in range(num_candidates)]
  res = res + '(constraint' + '(or ' + ' '.join(all_candidates) +  ')' + ')\n'

  for i in range(num_candidates):
    not_other_candidates = ' '.join(['(not c' + str(j+1) + ')' for j in range(num_candidates) if j != i])
    res = res + '(constraint (=> c' + str(i+1) + ' (and ' + not_other_candidates + ')))\n'
  return res


def writeSygusSynthConstraints(problem_dict,filename,write_or_append):
  num_quants = problem_dict['num-quants']
  formula = problem_dict['formula']
  train_imgs_list = problem_dict['trainimgs']
  candidate_imgs_list = problem_dict['candidateimgs']
  num_candidates = len(candidate_imgs_list)

  filehandle = open(filename,write_or_append)

  choose_one_candidate = genChooseOneCandidateConstraints(num_candidates)+'\n'
  filehandle.write(choose_one_candidate)

  for img in train_imgs_list:
    train_img_constraints = genOneImgConstraints(formula,num_quants,'t',img)
    filehandle.write(train_img_constraints)
  for i in range(len(candidate_imgs_list)):
    candidate_img_constraints = genOneImgConstraints(formula,num_quants,'c',candidate_imgs_list[i],i+1)
    filehandle.write(candidate_img_constraints)
    
  filehandle.close()


def writeSmtCheckConstraints(problem_dict,filename,write_or_append):
  #Constraints same as SyGuS but replaced with 'assert' instead of 'constraint'
  num_quants = problem_dict['num-quants']
  formula = problem_dict['formula']
  train_imgs_list = problem_dict['trainimgs']
  candidate_imgs_list = problem_dict['candidateimgs']
  num_candidates = len(candidate_imgs_list)

  filehandle = open(filename,write_or_append)

  choose_one_candidate = genChooseOneCandidateConstraints(num_candidates)+'\n'
  choose_one_candidate = choose_one_candidate.replace('constraint','assert')
  filehandle.write(choose_one_candidate)
  
  for img in train_imgs_list:
    train_img_constraints = genOneImgConstraints(formula,num_quants,'t',img)
    train_img_constraints = train_img_constraints.replace('constraint','assert')
    filehandle.write(train_img_constraints)
  for i in range(len(candidate_imgs_list)):
    candidate_img_constraints = genOneImgConstraints(formula,num_quants,'c',candidate_imgs_list[i],i+1)
    candidate_img_constraints = candidate_img_constraints.replace('constraint','assert')
    filehandle.write(candidate_img_constraints)
    
  filehandle.close()


def writeSmtSynthConstraints(problem_dict,filename,write_or_append):
  #Constraints same as SyGuS but replaced with 'assert' instead of 'constraint'
  #Currently using the guards as a funtion 'gi'. But might later change it to
  #actualy saying 'LabelOf objid synth-label' where the 'synth-label' is the
  #variable being asked to synthesise.
  num_quants = problem_dict['num-quants']
  formula = problem_dict['formula']
  train_imgs_list = problem_dict['trainimgs']
  candidate_imgs_list = problem_dict['candidateimgs']
  num_candidates = len(candidate_imgs_list)

  filehandle = open(filename,write_or_append)

  choose_one_candidate = genChooseOneCandidateConstraints(num_candidates)+'\n'
  choose_one_candidate = choose_one_candidate.replace('constraint','assert')
  filehandle.write(choose_one_candidate)
  
  for img in train_imgs_list:
    train_img_constraints = genOneImgConstraints(formula,num_quants,'t',img)
    train_img_constraints = train_img_constraints.replace('constraint','assert')
    filehandle.write(train_img_constraints)
  for i in range(len(candidate_imgs_list)):
    candidate_img_constraints = genOneImgConstraints(formula,num_quants,'c',candidate_imgs_list[i],i+1)
    candidate_img_constraints = candidate_img_constraints.replace('constraint','assert')
    filehandle.write(candidate_img_constraints)
    
  filehandle.close()