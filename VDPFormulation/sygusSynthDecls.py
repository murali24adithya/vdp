from vdputils import *

def genQuantDeclsSygusSynth(problem_dict):
  num_quants = problem_dict['num-quants']
  res =''
  for i in range(num_quants):
    res = res + '''
(synth-fun q{0} () Bool
  ((Start Bool))
  ((Start Bool 
      (true false))
))\n'''.format(str(i+1))
  return res


def genGuardDeclsSygusSynth(problem_dict):
  num_quants = problem_dict['num-quants']
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(label_datatype)
  res = ''
  for i in range(num_quants):
    res = res + '''
(synth-fun g{0} ((x {1})) Bool
((Start Bool) (Lb {2}) (Ob {1}))
((Start Bool
		((labelOf Ob Lb)))
 (Lb {2}
	 	({3}))
 (Ob {1}
	 	(x))
))\n'''.format(str(i+1),obj_datatype,label_datatype,' '.join(label_set))
  return res


def genGrammarDeclSygusSynth(problem_dict):
  num_quants = problem_dict['num-quants']
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  grammar_choice = problem_dict['grammar-flag']
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(label_datatype)
  res = ''
  var_decl = ''
  for i in range(num_quants):
    var_decl = var_decl + '(x' + str(i+1) + ' {0})'.format(obj_datatype)
  var_list = ['x'+str(i+1) for i in range(num_quants)]  
  res = res + '''
(synth-fun f ({0}) Bool
((B Bool) (Lb {2}) (Ob {1}))
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
(Lb {2}
	 	 ({3}))
(Ob {1}
	 	 ({4}))
	))\n'''.format(var_decl,obj_datatype,label_datatype,' '.join(label_set),' '.join(var_list))
  return res


def genCandidateDeclsSygusSynth(problem_dict):
  num_candidates = len(problem_dict['candidateimgs'])
  res = ''
  for i in range(num_candidates):
    res = res + '''
(synth-fun c{} () Bool
  ((Start Bool))
  ((Start Bool 
        (true false))
))'''.format(str(i+1))
  return res


def genSygusSynthDecls(problem_dict):
  res = ''
  
  #Quantifier declarations
  res = genQuantDeclsSygusSynth(problem_dict) 
  res = res + '\n'
  
  #Guard declarations
  res = res + genGuardDeclsSygusSynth(problem_dict)
  res = res + '\n'

  #Grammar declaration
  res = res + genGrammarDeclSygusSynth(problem_dict)
  res = res + '\n'

  #Candidate declaration
  res = res + genCandidateDeclsSygusSynth(problem_dict)
  res = res + '\n'

  return res


def writeSygusSynthDecls(problem_dict,filename,write_or_append):
  decls = genSygusSynthDecls(problem_dict)
  filehandle = open(filename,write_or_append)
  filehandle.write(decls)
  filehandle.close()