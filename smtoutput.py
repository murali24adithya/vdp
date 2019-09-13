from parsetree import *
import sys

# Consider implementing in a more general fashion using regexes
# Or interfacing temporarily with Z3Py
# Extend to include when variables are not present in smt model

opstr = lambda i: '  (define-fun op' + str(i) + ' () Ops'
bindstr = lambda i,j: '  (define-fun arg' + str(i) + '_' + str(j) + ' () Binding'
negstr = lambda i: '  (define-fun n' + str(i) + ' () IfNeg'
relstr = lambda i: '  (define-fun r' + str(i) + ' () Rels'
quantstr = lambda i: '  (define-fun q' + str(i) + ' () Quantifier'
    
oplookup = lambda i,lst: lst[lst.index(opstr(i))+1].split(' ')[-1][:-1]
bindlookup = lambda i,j,lst: lst[lst.index(bindstr(i,j))+1].split(' ')[-1][:-1]
neglookup = lambda i,lst: lst[lst.index(negstr(i))+1].split(' ')[-1][:-1]
rellookup = lambda i,lst: lst[lst.index(relstr(i))+1].split(' ')[-1][:-1]
quantlookup = lambda i,lst: lst[lst.index(quantstr(i))+1].split(' ')[-1][:-1]



def recover_baseformula(lst,dt,arity):
  result = ""
  baseformula_dict = {}

  if type(dt) == list:
    opi = "op" + str(dt[0])
    op_i = oplookup(dt[0],lst)
    baseformula_dict[opi] = op_i

    (baseformula_left_dict,baseformula_left) = recover_baseformula(lst,dt[1],arity)
    (baseformula_right_dict, baseformula_right) = recover_baseformula(lst, dt[2], arity)

    baseformula_dict.update(baseformula_left_dict)
    baseformula_dict.update(baseformula_right_dict)

    result = result + '(' + op_i[2:] +'\n' + baseformula_left + '\n' + baseformula_right + ')'
    return (baseformula_dict,result)

  else:
    ri = "r" + str(dt)
    r_i = rellookup(dt,lst)
    ni = "n" + str(dt)
    n_i = neglookup(dt,lst)

    baseformula_dict[ri] = r_i
    baseformula_dict[ni] = n_i

    for j in range(1,arity+1):
      argij = "arg" + str(dt) + "_" + str(j)
      argi_j = bindlookup(dt,j,lst)
      baseformula_dict[argij] = argi_j
      result = result + argi_j[5:] + ","
    
    result = r_i + "(" + result[:-1] + ")"
    if n_i == 'yes':
      result = "Not " + result
    return (baseformula_dict,"[" + result + "]")
  


def construct_formula(model,treearg,num_vars,arity):
  #print model
  if type(treearg) == int:
    pt = gen_default_parsetree(treearg)
  elif type(treearg) == list:
    pt = treearg
  else:
    raise ValueError('You must specify either a number of relations or a parsetree.')

  dt = decorate_parsetree(pt)[0]

  result = ""
  formula_dict = {}

  for i in range(1, num_vars + 1):
    quant_i = quantlookup(i, model)
    qi = "q" + str(i)
    formula_dict[qi] = quant_i
    if quant_i == 'all':
      result = result + 'Forall x' + str(i) + '. '
    else:  # quant_i == 'one':
      result = result + 'Exists x' + str(i) + '. '

  (baseformula_dict,baseformula) = recover_baseformula(model, dt, arity)
  formula_dict.update(baseformula_dict)

  result = result + "\n" + baseformula
  #print baseformula
  return (formula_dict,result)


# def read_smt_output(smtfile,treearg,num_vars,arity):
#   result = ""
#
#   smt_file = open(smtfile, 'r')
#   smtstr = smt_file.read()
#
#   results = smtstr.split('Results for ')[1:]
#
#   for each_res in results:
#     each_res = each_res.split('\n')
#     result = result + "Results for test image " + each_res[0][1] + "\n"
#
#     if 'unsat' in each_res[1]:
#       result = result + 'Unsat: cannot be a solution\n'
#     else:
#       result = result + construct_formula(each_res[2:],treearg,num_vars,arity) + "\n"
#     result = result + "++++++++++++++\n"
#
#   smt_file.close()
#
#   return result


def read_single_smt_output(smtfile, treearg, num_vars, arity):
  smt_file = open(smtfile, 'r')
  smtstr = smt_file.readline()

  if 'unsat' in smtstr:
    smt_file.close()
    return ('unsat',{},"")
  else:
    smtstr = smt_file.read().split('\n')
    (formula_dict,pretty_formula) = construct_formula(smtstr,treearg,num_vars,arity)
    pretty_formula = pretty_formula + "\n++++++++++++++\n"

    result = ('sat',formula_dict,pretty_formula)
    smt_file.close()
    return result




# smt_outfile = sys.argv[1]
# num_rels = int(sys.argv[2])
# num_vars = int(sys.argv[3])
# arity = 2
#
# read_smt_output(smt_outfile,num_rels,num_vars,arity)