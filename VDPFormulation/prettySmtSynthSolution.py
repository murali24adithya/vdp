import sys

def getValuationPair(smt_valuation):
  #Currently only supporting valuations that are boolean.
  ##Example1: (define-fun labelOf_x1_sofa () Bool false)
  ##Example2: (define-fun lbl_g1 () (_ BitVec 2) (_ bv0 2))
  ##Example3: (define-fun constval_chair () (_ BitVec 2) (_ bv0 2))
  val_str = smt_valuation.split(' ')
  variable = val_str[1]
  if variable.startswith('lbl_g') or variable.startswith('constval_'):
    value = val_str[-2][-1]
  else:
    value = val_str[-1][:-1]
  return (variable,value)


def getSoldict(filename):
  filehandle = open(filename,'r')
  s = filehandle.read()
  s = s.split('\n')
  if s[0] == 'unsat':
    print('Unsat. No model available.')
    return None
  elif s[0] == 'sat':
    s = s[2:-2]
    soldict = {}
    for smt_valuation in s:
      (variable,value) = getValuationPair(smt_valuation)
      soldict[variable] = value
    return soldict
  else:
    raise ValueError('Something is wrong. Solver should either return sat or unsat as the first line.')


def recoverFormula(soldict):
  variables = soldict.keys()
  quant_cond = lambda x: x.startswith('q')
  guard_cond = lambda x: x.startswith('lbl_g')
  label_cond = lambda x: x.startswith('constval_')
  candidate_cond = lambda x: x.startswith('c')

  quantifier_vars = [x for x in variables if quant_cond(x)]
  guard_vars = [x for x in variables if guard_cond(x)]
  label_identifier_vars = [x for x in variables if label_cond(x)]
  candidate_vars = [x for x in variables if candidate_cond(x) and not label_cond(x)]
  rel_vars = [x for x in variables if not (quant_cond(x) or guard_cond(x) or label_cond(x) or candidate_cond(x))]

  quant_guard_list = []
  for i in range(len(quantifier_vars)):
    quant_val = soldict[quantifier_vars[i]]
    guard_val = [label_identifier for label_identifier in label_identifier_vars if soldict[label_identifier] == soldict['lbl_g'+str(i+1)]][0]
    quant_guard_list = quant_guard_list + [(quant_val,guard_val)]
  
  #print(quant_guard_list)

  formula_rels = [rel_var for rel_var in rel_vars if soldict[rel_var] == 'true']

  #print(formula_rels)

  candidate = [candidate_var for candidate_var in candidate_vars if soldict[candidate_var] == 'true'][0]

  #print(candidate)

  return [quant_guard_list,formula_rels,candidate]

  
def printFormula(raw_formula):
  res = ''

  [quant_guards,formula,candidate] = raw_formula
  for i in range(len(quant_guards)):
    (quant,guard) = quant_guards[i]
    if quant == 'true':
      quant = 'Forall'
    elif quant == 'false':
      quant = 'Exists'
    else:
      raise ValueError('Something is wrong. Quantifier value should either be true or false.')
    #Remove the 'constval_' prefix from the guard
    guard = guard[9:]
    res = res + '{0} x{1}: {2}. '.format(quant,str(i+1),guard)
  
  res = res + '(AND {0})\n'.format(' '.join(formula))

  res = res + 'Candidate {0}'.format(candidate[1:])

  return res

filename = 'in.txt'
filename = sys.argv[1]

soldict = getSoldict(filename)

# for key in soldict.keys():
#   print((key,soldict[key]))

if soldict != None:
  raw_formula = recoverFormula(soldict)
  print(printFormula(raw_formula))


