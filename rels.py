#Machinery for evaluating relations simply
#Will also help define more relations if necessary
#Needs the input dictionary to be organised in a certain way for this to work. 

def define_fun_label(d):
  temp = "nullobj"
  label_dict = d['label']
  objs = label_dict.keys()
  for obj in objs:
    temp = "(ite (= x " + obj + ") "+ label_dict[obj] +"\n" + temp + ")"
  
  result = "(define-fun label ((x Obj)) Obj\n" + temp + "\n)"

  preamble = ";Machinery for looking up labels of objects\n" + ";Default value is 'nullobj'\n"
  return preamble + result

# Defines the function that evaluates each object to its corresponding boundary. One function for each boundary.
def define_fun_border(d,direction):
  if direction not in ['top', 'bottom', 'left', 'right']:
    raise ValueError('Direction must be one of top, bottom, left or right')

  temp = " 0"
  border_dict = d[direction + '_border']
  objs = border_dict.keys()
  for obj in objs:
    temp = "(ite (= x " + obj + ") "+ str(border_dict[obj]) +"\n" + temp + ")"
  
  result = "(define-fun " + direction + "_border ((x Obj)) Int\n" + temp + "\n)"
  
  preamble = ";Machinery for looking up the " + direction + " boundary of the bounding boxes\n" + ";Default value is 0\n"
  return preamble + result

# Defines the datatype of relations
def declare_datatype_rels(counting='no'):
  preamble = ";Datatype of relations--made into a uniform arity\n"
  if counting == 'yes':
    relstr = " grCount lsCount eqCount"
  else:
    relstr = ""
  result = "(declare-datatypes () ((Rels neq labelOf sameLabel toLeft toRight above below within" + relstr + ")))"
  return preamble + result

# Auxiliary function for universal relation evaluator generation
def temp_neq_gen():
  temp_neq = "(and (= r neq) (not (= b a)))"
  return temp_neq

# # Auxiliary function for universal relation evaluator generation
# def temp_labelOf_gen():
#   temp_labelOf = "(and (= r labelOf) (= b (label a)))"
#   return temp_labelOf

# Auxiliary function for universal relation evaluator generation
# Support for multiple labels
def temp_labelOf_gen(d):
  imgdict = {}
  imgdict.update(d['train'])
  imgdict.update(d['test'])

  result = ""

  imgs = imgdict.keys()
  for img in imgs:
    objs = imgdict[img].keys()
    for obj in objs:
      labels = imgdict[img][obj][0]
      for label in labels:
        result = result + "(and (= r labelOf) (= a " + obj + ") (= b " + label + "))\n"

  return result


# Auxiliary function for universal relation evaluator generation
def temp_sameLabel_gen():
  temp_sameLabel = "(and (= r sameLabel) (= (label b) (label a)))"
  return temp_sameLabel

# Auxiliary function for universal relation evaluator generation
def temp_toLeft_gen(d):
  temp_toLeft = "(and (= r toLeft) (< (right_border a) (left_border b)))"
  return temp_toLeft

# Auxiliary function for universal relation evaluator generation
def temp_toRight_gen(d):
  temp_toRight = "(and (= r toRight) (> (left_border a) (right_border b)))"
  return temp_toRight

# Auxiliary function for universal relation evaluator generation
def temp_above_gen(d):
  temp_above = "(and (= r above) (< (bottom_border a) (top_border b)))"
  return temp_above

# Auxiliary function for universal relation evaluator generation
def temp_below_gen(d):
  temp_below = "(and (= r below) (> (top_border a) (bottom_border b)))"
  return temp_below

# Auxiliary function for universal relation evaluator generation
def temp_within_gen(d):
  temp_within_top = " (> (top_border a) (top_border b))"
  temp_within_bottom = " (< (bottom_border a) (bottom_border b))"
  temp_within_left = " (> (left_border a) (left_border b))"
  temp_within_right = " (< (right_border a) (right_border b))"

  temp_within = "(and (= r within) (and\n" + temp_within_top + "\n" + temp_within_bottom + "\n" + temp_within_left + "\n" + temp_within_right + "))"

  return temp_within


# Defines the universal relation evaluator
# Currently does not include count of objects with the same label.
# Supported relations are manually defined. All arities are 2.
def define_fun_re(d):
  temp_neq = temp_neq_gen()
  temp_labelOf = temp_labelOf_gen()
  temp_sameLabel = temp_sameLabel_gen()
  temp_toLeft = temp_toLeft_gen(d)
  temp_toRight = temp_toRight_gen(d)
  temp_above = temp_above_gen(d)
  temp_below = temp_below_gen(d)
  temp_within = temp_within_gen(d)

  temp = "(or" + ("\n" + temp_neq) + ("\n" + temp_labelOf) + ("\n" + temp_sameLabel) + ("\n" + temp_toLeft) + ("\n" + temp_toRight) + ("\n" + temp_above) + ("\n" + temp_below) + ("\n" + temp_within) + "\n)"
  result = "(define-fun re ((r Rels)(a Obj)(b Obj)) Bool\n" + temp +")"

  preamble = ";Relation evaluator\n" + ";All relations are made into a uniform arity using 'nullobj' as a default argument\n"
  return preamble + result