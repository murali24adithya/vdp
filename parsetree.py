# Generates a default parsetree given number of leaves desired
# Splits into halves, breaking ties to the right
def gen_default_parsetree(num_rels):
  if num_rels == 1:
    return 0
  else:
    half = num_rels/2
    left = gen_default_parsetree(half)
    right = gen_default_parsetree(num_rels-half)
    return [left,right]

#Annotates a parsetree 't' with relation numbers and operator numbers
# returns (result,rel_count,op_count)
# result is the desired annotated parse tree. Each level is a triple, denoting the operator number and the two operands. Leaves are relaton numbers
# rel_count and op_count and continuing counts in an in-order traversal
def decorate_parsetree(t,rel_count=1,op_count=1):
  if type(t) != list:
    return (rel_count,rel_count+1,op_count)
  else:
    if len(t) != 2:
      raise ValueError('Parse tree not binary')
    else:
      (t_left,rc,oc) = decorate_parsetree(t[0],rel_count,op_count)
      (t_right,full_rc,full_oc) = decorate_parsetree(t[1],rc,oc+1)
      return ([oc,t_left,t_right],full_rc,full_oc)
# dec_ps_test = [0,[[0,0],0]]


# Machinery to generate the evaluation of abstract relation expressions in the base formula
def rel_expr(index,num_vars,arity,counting='no'):
  varstr = ""
  for i in range(1,num_vars+1):
    varstr = varstr + ("x" + str(i)) + " "
  
  temp = ""
  for i in range(1,arity+1):
    temp = temp + " (bindlook " + varstr + ("arg" + str(index) + "_" + str(i) + ")")

  if counting == 'yes':
    ifImg = "img"
  else:
    ifImg = ""

  result = "(negeval n" + str(index) + " (re r" + str(index) + temp + " " + ifImg + "))"
  return result


# Fills a given decorated parsetree 'dt' with the appropriate operator variables+evaluations and relation variables+evaluations.
def annotate_parsetree(dt,num_vars,arity,counting='no'):
  if type(dt) == tuple:
    raise ValueError('You gave me the emtire tuple. Give me just the tree.')

  if type(dt) != list:
    return rel_expr(dt,num_vars,arity,counting)
  else:
    return ("(opeval op" + str(dt[0])) + "\n" + annotate_parsetree(dt[1],num_vars,arity) + "\n" + annotate_parsetree(dt[2],num_vars,arity) + ")"