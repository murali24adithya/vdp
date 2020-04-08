import itertools
import sys
import string

# def processNNOutputOneImage(raw_img,imgid,mode):
#   if mode == 't':
#     img_tag = 't'
#   elif mode == 'c':
#     img_tag = 'c'
#   else:
#     raise ValueError('Incorrect mode')

#   s = raw_img.split('\n')[1:-1]
#   img_list = []
#   i = 0
#   while i < len(s):
#     raw_label = s[i]
#     raw_bounding_box = s[i+1]
    
#     objdict = {}
#     label = raw_label.split(':')[0]
#     left_margin = raw_bounding_box.split(':')[1].split(',')[0].split('=')[1]
#     top_margin = raw_bounding_box.split(':')[1].split(',')[1].split('=')[1]
#     right_margin = raw_bounding_box.split(':')[1].split(',')[2].split('=')[1]
#     bottom_margin = raw_bounding_box.split(':')[1].split(',')[3].split('=')[1]

#     objdict['label'] = label
#     objdict['left'] = int(left_margin)
#     objdict['top'] = int(top_margin)
#     objdict['right'] = int(right_margin)
#     objdict['bottom'] = int(bottom_margin)
#     objdict['objid'] = img_tag + str(imgid) + 'o' + str(i//2 + 1)
#     objdict['imgid'] = mode + str(imgid)
#     img_list = img_list + [objdict]

#     i = i+2

#   return img_list


def processNNOutputOneImage(raw_img,imgid,mode):
  if mode == 't':
    img_tag = 't'
  elif mode == 'c':
    img_tag = 'c'
  else:
    raise ValueError('Incorrect mode')
  
  
  attributes = raw_img.split('\n')[1:-1]
  #Image representation will be a list of object dictionaries
  img = []

  running_objid = 1
  objdict = {'labels':[]}
  for attribute in attributes:
    components = attribute.split(': ')
    if components[0] != 'Bounding Box':
      #Has to be one of the labels of the object
      label = components[0]
      #Transform label to handle multi-word labels like 'teddy bear' instead of 'cat'
      label = '_'.join(label.split(' '))
      objdict['labels'] = objdict['labels'] + [label]

      #Not incorporating confidence percentages in the object representation at this point
      confidence = int(components[1][:-1])

    else:
      #Bounding box. If there are multiple ones, the first one will be chosen
      #Support has to be better for this. The logic could go wrong if there
      ##are boxes without labels, labels without boxes, etc. Solution is to
      ## modify YOLO to give more structured output
      boundaries = components[1]
      left_margin = boundaries.split(', ')[0].split('=')[1]
      top_margin = boundaries.split(', ')[1].split('=')[1]
      right_margin = boundaries.split(', ')[2].split('=')[1]
      bottom_margin = boundaries.split(', ')[3].split('=')[1]
      
      objdict['left'] = int(left_margin)
      objdict['top'] = int(top_margin)
      objdict['right'] = int(right_margin)
      objdict['bottom'] = int(bottom_margin)

      #Construcut other attributes of the object and add the dictionary to the list
      objdict['objid'] = img_tag + str(imgid) + 'o' + str(running_objid)
      objdict['imgid'] = img_tag + str(imgid)
      img = img + [objdict]
      
      #Update running object id counter and refresh object dictionary
      running_objid = running_objid + 1
      objdict = {'labels':[]}

  return img



def processNNOutputManyImages(name,mode):
  filehandle = open(name,'r')
  s = filehandle.read()
  s = s.split('Enter')[1:-1]
  list_of_imgs = []
  for i in range(len(s)):
    list_of_imgs = list_of_imgs + [processNNOutputOneImage(s[i],i+1,mode)]
  filehandle.close()
  return list_of_imgs



def genObjDatatype(all_imgs_list):
  res = ''
  objid_list = [objdict['objid'] for img in all_imgs_list for objdict in img]
  for objid in objid_list:
    res = res + '(' + objid + ')' 
  return '(declare-datatypes ((Obj 0))\n((\n' + res + '\n)))\n\n' 


def genLabelDatatype(all_imgs_list):
  res = ''
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  for label in label_set:
    res = res + '(' + label + ')' 
  return '(declare-datatypes ((Label 0))\n((\n' + res + '\n)))\n\n' 


def genRels(all_imgs_list,relname):
  if relname == 'labelOf':
    res = ''
    for img in all_imgs_list:
      for objdict in img:
        for label in objdict['labels']:
          res = res + '(and (= obj ' + objdict['objid'] + ') (= lbl ' + label + '))\n'
    return '(define-fun labelOf((obj Obj)(lbl Label)) Bool\n(or\n' + res + '\n))\n\n'
  
  elif relname in ['leftOf','rightOf','above','below','within']:
    if relname == 'leftOf': 
      relFunc = lambda x,y: x['right'] <= y['left'] and x != y
    elif relname == 'rightOf':
      relFunc = lambda x,y: x['left'] >= y['right'] and x != y
    elif relname == 'above':
      relFunc = lambda x,y: x['bottom'] <= y['top'] and x != y
    elif relname == 'below':
      relFunc = lambda x,y: x['top'] >= y['bottom'] and x != y
    else: #rel == 'within'
      relFunc = lambda x,y: x['left'] >= y['left'] and x['top'] >= y['top'] and x['right'] <= y['right'] and x['bottom'] <= y['bottom'] and x != y
    
    rel_pairs = []
    for img in all_imgs_list:
      rel_pairs = rel_pairs + [(x['objid'],y['objid']) for x in img for y in img if relFunc(x,y)]
    if rel_pairs == []:
      #No pairs satisfy the relation
      res = 'false'
    else:
      res = ''
      for rel_pair in rel_pairs:
        res = res + '(and (= x ' + rel_pair[0] + ') (= y ' + rel_pair[1] + '))\n'
    return '(define-fun ' + relname + '((x Obj)(y Obj)) Bool\n(or\n' + res + '\n))\n\n'

  else:
    raise ValueError('Incorrect relation symbol.')



def writeDatatypesAndRels(problem_dict,filename,write_or_append):
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  rels = ['labelOf','leftOf','rightOf','above','below','within']

  filehandle = open(filename,write_or_append)
  filehandle.write(genObjDatatype(all_imgs_list))
  filehandle.write(genLabelDatatype(all_imgs_list))
  for rel in rels:
    filehandle.write(genRels(all_imgs_list,rel))
  filehandle.close()



def subst(formula,orig_term,new_term):
  if formula == orig_term:
    return new_term
  elif not isinstance(formula,list):
    return formula
  else:
    return [new_term if subformula == orig_term else subst(subformula,orig_term,new_term) for subformula in formula]


def prettyFormula(formula):
  if not isinstance(formula,list):
    return formula + '\n'
  elif all([not isinstance(subformula,list) for subformula in formula]):
    return '(' + ' '.join(formula) + ')\n'
  else:
    return '(' + ''.join([prettyFormula(subformula) for subformula in formula]) + ')\n'



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


def genOneImgConstraints(formula,num_quants,mode,img,imgindex=None):
  objs_in_img = [objdict['objid'] for objdict in img]

  if mode == 't':
    constraints = genAllQuantsTrueConstraints(formula,objs_in_img,num_quants)
    return '(constraint\n' + prettyFormula(constraints) + ')\n\n'

  elif mode == 'c':
    true_constraints = genAllQuantsTrueConstraints(formula,objs_in_img,num_quants)
    true_constraints = ['=>',['=','c'+str(imgindex),'true'],true_constraints]
    true_constraints_string = '(constraint\n' + prettyFormula(true_constraints) + ')\n\n'

    false_constraints = genAllQuantsFalseConstraints(formula,objs_in_img,num_quants)
    false_constraints = ['=>',['=','c'+str(imgindex),'false'],false_constraints]
    false_constraints_string = '(constraint\n' + prettyFormula(false_constraints) + ')\n\n'

    return true_constraints_string + false_constraints_string

  else:
    raise ValueError('Incorrect mode.')


def genChooseOneCandidateConstraints(num_candidates):
  res = ''
  all_candidates = ['c'+str(i+1) for i in range(num_candidates)]
  res = res + '(constraint' + '(or ' + ' '.join(all_candidates) +  ')' + ')\n'

  for i in range(num_candidates):
    not_other_candidates = ' '.join(['(not c' + str(j+1) + ')' for j in range(num_candidates) if j != i])
    res = res + '(constraint (=> c' + str(i+1) + ' (and ' + not_other_candidates + ')))\n'
  return res


def writeSynthConstraints(problem_dict,filename,write_or_append):
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


def writeSmtConstraints(problem_dict,filename,write_or_append):
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

def genQuantDeclsSynth(num_quants):
  res =''
  for i in range(num_quants):
    res = res + '''
(synth-fun q{0} () Bool
  ((Start Bool))
  ((Start Bool 
      (true false))
))\n'''.format(str(i+1))
  return res


def genGuardDeclsSynth(num_quants,label_set):
  res = ''
  for i in range(num_quants):
    res = res + '''
(synth-fun g{0} ((x Obj)) Bool
((Start Bool) (Lb Label) (Ob Obj))
((Start Bool
		((labelOf Ob Lb)))
 (Lb Label
	 	({1}))
 (Ob Obj
	 	(x))
))\n'''.format(str(i+1),' '.join(label_set))
  return res


def genGrammarDeclSynth(grammar_choice,num_quants,label_set):
  res = ''
  var_decl = ''
  for i in range(num_quants):
    var_decl = var_decl + '(x' + str(i+1) + ' Obj)'
  var_list = ['x'+str(i+1) for i in range(num_quants)]  
  res = res + '''
(synth-fun f ({0}) Bool
((B Bool) (Lb Label) (Ob Obj))
((B Bool
		(true
		 false
		 (= Ob Ob)
		 (=> B B)
		 (and B B)
		 (or B B)
		 (not B)
		 (labelOf Ob Lb)
		 (leftOf Ob Ob) (rightOf Ob Ob) (above Ob Ob) (below Ob Ob) (within Ob Ob)))
(Lb Label
	 	 ({1}))
(Ob Obj
	 	 ({2}))
	))\n'''.format(var_decl,' '.join(label_set),' '.join(var_list))
  return res


def genCandidateDeclsSynth(num_candidates):
  res = ''
  for i in range(num_candidates):
    res = res + '''
(synth-fun c{} () Bool
  ((Start Bool))
  ((Start Bool 
        (true false))
))'''.format(str(i+1))
  return res


def genSynthDecls(problem_dict):
  res = ''
  
  #Quantifier declarations
  num_quants = problem_dict['num-quants']
  res = genQuantDeclsSynth(num_quants) 
  res = res + '\n'
  
  #Guard declarations
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  res = res + genGuardDeclsSynth(num_quants,label_set)
  res = res + '\n'

  #Grammar declaration
  grammar_choice = problem_dict['grammar-flag']
  res = res + genGrammarDeclSynth(grammar_choice,num_quants,label_set)
  res = res + '\n'

  #Candidate declaration
  num_candidates = len(problem_dict['candidateimgs'])
  res = res + genCandidateDeclsSynth(num_candidates)
  res = res + '\n'

  return res


def writeSynthDecls(problem_dict,filename,write_or_append):
  decls = genSynthDecls(problem_dict)
  filehandle = open(filename,write_or_append)
  filehandle.write(decls)
  filehandle.close()


def genSmtDecls(problem_dict):
  res = ''
  
  #Quantifier declarations
  num_quants = problem_dict['num-quants']
  for i in range(num_quants):
    res = res + '(declare-const q{0} Bool)\n'.format(str(i+1)) 
    res = res + '(assert (= q{0} {1}))\n'.format(str(i+1),problem_dict['quantifier-solution'][i]) 
  res = res + '\n'
  
  #Guard declarations
  for i in range(num_quants):
    res = res + '(define-fun g{0} ((x Obj)) Bool\n{1}\n)\n'.format(str(i+1),problem_dict['guard-solution'][i])
  res = res + '\n'

  #Inner formula declaration
  var_decl = ''
  for i in range(num_quants):
    var_decl = var_decl + '(x' + str(i+1) + ' Obj)'
  res = res + '(define-fun f ({0}) Bool\n{1}\n)\n'.format(var_decl,problem_dict['qf-formula-solution'])
  res = res + '\n'

  #Candidate flag declaration
  num_candidates = len(problem_dict['candidateimgs'])
  for i in range(num_candidates):
    res = res + '(declare-const c{0} Bool)\n'.format(str(i+1))
  res = res + '(assert (= c{0} true))\n'.format(problem_dict['candidate-solution'])
  res = res + '\n'
  
  res = res + '\n'

  return res

def writeSmtDecls(problem_dict,filename,write_or_append):
  decls = genSmtDecls(problem_dict)
  filehandle = open(filename,write_or_append)
  filehandle.write(decls)
  filehandle.close()  


def generateSygusFile(problem_dict):
  instance = problem_dict['instance']
  synth_filename = instance + '_sygus_file.sy'
  filehandle = open(synth_filename,'w')
  filehandle.write('(set-logic ALL)\n')
  filehandle.close()
  writeDatatypesAndRels(problem_dict,synth_filename,'a')
  writeSynthDecls(problem_dict,synth_filename,'a')
  writeSynthConstraints(problem_dict,synth_filename,'a')
  filehandle = open(synth_filename,'a')
  filehandle.write('\n(check-synth)')
  filehandle.close()


def generateSmtFile(problem_dict):
  instance = problem_dict['instance']
  smt_filename = instance + '_smt_file.smt'
  filehandle = open(smt_filename,'w')
  filehandle.write('(set-logic ALL)\n')
  filehandle.close()
  #Datatypes and Rels same as SyGuS file
  writeDatatypesAndRels(problem_dict,smt_filename,'a')
  #New function for SMT Decls. Basically SyGuS decls with empty bodies
  writeSmtDecls(problem_dict,smt_filename,'a')
  #Constraints same as SyGuS but replaced with 'assert' instead of 'constraint'
  writeSmtConstraints(problem_dict,smt_filename,'a')
  filehandle = open(smt_filename,'a')
  filehandle.write('\n(check-sat)')
  filehandle.close()  



def genProblemDict(instance,num_quants):
  trainfile_name = instance + '_train_out.txt'
  candidatefile_name = instance + '_test_out.txt'
  train_imgs_list = processNNOutputManyImages(trainfile_name,'t')
  candidate_imgs_list = processNNOutputManyImages(candidatefile_name,'c')
  #all_imgs_list = train_imgs_list + candidate_imgs_list
  
  vars_list = ['x'+str(i+1) for i in range(num_quants)]
  formula = ['f'] + vars_list
  
  problem_dict = {}
  problem_dict['instance'] = instance
  problem_dict['formula'] = formula
  problem_dict['num-quants'] = num_quants
  problem_dict['trainimgs'] = train_imgs_list[1:]
  problem_dict['candidateimgs'] = candidate_imgs_list
  problem_dict['grammar-flag'] = ''

  return problem_dict




#instance = 'adoc'
#num_quants = 2
#sygus_or_smt = 'smt'
instance = sys.argv[1]
num_quants = int(sys.argv[2])
sygus_or_smt = sys.argv[3]
problem_dict = genProblemDict(instance,num_quants)
if sygus_or_smt == 'sygus':
  generateSygusFile(problem_dict)
elif sygus_or_smt == 'smt':
  problem_dict['quantifier-solution'] = ['' for i in range(num_quants)]
  problem_dict['guard-solution'] = ['' for i in range(num_quants)]
  problem_dict['qf-formula-solution'] = ''
  problem_dict['candidate-solution'] = ''
  generateSmtFile(problem_dict)
else:
  raise ValueError('Specify either sygus or smt for which file to generate.')


#writeDatatypesAndRels(problem_dict,'rels.txt','w')
#writeSynthDecls(problem_dict,'synthdecls.txt','w')
#writeConstraints(problem_dict,'constraints.txt','w')


