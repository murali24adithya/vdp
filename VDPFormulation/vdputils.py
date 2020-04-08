def makeDatatype(raw_datatype):
  #Currently only supporting BitVec type and enumerative datatypes
  if raw_datatype[0] == 'BitVec':
    datatype = '(_ BitVec ' + str(raw_datatype[1]) + ')'
  else:
    datatype = raw_datatype[0]
  return datatype


def convertConstant(const,raw_datatype):
  #Currently only suporting BitVec type and integer constants
  return '(_ bv' + str(const) + ' ' +  str(raw_datatype[1]) + ')'



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
