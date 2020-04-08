from vdputils import *

def genSmtCheckDecls(problem_dict):
  num_quants = problem_dict['num-quants']
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  
  res = ''
  
  #Quantifier declarations
  for i in range(num_quants):
    res = res + '(declare-const q{0} Bool)\n'.format(str(i+1)) 
    res = res + '(assert (= q{0} {1}))\n'.format(str(i+1),problem_dict['quantifier-solution'][i]) 
  res = res + '\n'
  
  #Guard declarations
  for i in range(num_quants):
    res = res + '(define-fun g{0} ((x {1})) Bool\n{2}\n)\n'.format(str(i+1),obj_datatype,problem_dict['guard-solution'][i])
  res = res + '\n'

  #Inner formula declaration
  var_decl = ''
  for i in range(num_quants):
    var_decl = var_decl + '(x' + str(i+1) + ' {0})'.format(obj_datatype)
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

def writeSmtCheckDecls(problem_dict,filename,write_or_append):
  decls = genSmtCheckDecls(problem_dict)
  filehandle = open(filename,write_or_append)
  filehandle.write(decls)
  filehandle.close()