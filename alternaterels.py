from rels import *
import itertools

# lt top rt bot
def precompute_gen(d,rel):
  if rel == 'sameLabel':
    func = lambda x, y: not set(x[0]).isdisjoint(y[0])
  elif rel == 'toLeft':
    func = lambda x, y: x[2]['right'] < y[2]['left']
  elif rel == 'toRight':
    func = lambda x, y: x[2]['left'] > y[2]['right']
  elif rel == 'above':
    func = lambda x, y: x[2]['bottom'] < y[2]['top']
  elif rel == 'below':
    func = lambda x, y: x[2]['top'] > y[2]['bottom']
  elif rel == 'within':
    func = lambda x, y: (x[2]['left'] > y[2]['left']) and (x[2]['top'] > y[2]['top']) and (x[2]['right'] < y[2]['right']) and (x[2]['bottom'] < y[2]['bottom'])
  else:
    raise ValueError('Cannot precompute relation ' + str(rel))
  
  imgdict = {}
  imgdict.update(d['train'])
  imgdict.update(d['test'])

  result = ""

  imgs = imgdict.keys()
  for img in imgs:
    objs = imgdict[img].keys()

    for obj1 in objs:
      for obj2 in objs:
        if func(imgdict[img][obj1],imgdict[img][obj2]):
          result = result + "(and (= r " + rel + ") (= a " + obj1 + ") (= b " + obj2 + "))\n"
  
  return result


def make_temp_counts(d):
  lbldict = d['labelcount']
  result = ""
  imgs = lbldict.keys()
  for img in imgs:
    lbls = lbldict[img]
    for lbl1 in lbls:
      for lbl2 in lbls:
        if lbldict[img][lbl1] > lbldict[img][lbl2]:
          result = result + "(and (= r grCount) (= a " + lbl1 + ") (= b " + lbl2 + ") (= img " + img + "))\n"
        elif lbldict[img][lbl1] < lbldict[img][lbl2]:
          result = result + "(and (= r lsCount) (= a " + lbl1 + ") (= b " + lbl2 + ") (= img " + img + "))\n"
        elif lbldict[img][lbl1] == lbldict[img][lbl2]:
          result = result + "(and (= r eqCount) (= a " + lbl1 + ") (= b " + lbl2 + ") (= img " + img + "))\n"

  return result



def define_fun_precompute_re(d,counting='no'):
  temp_neq = temp_neq_gen()
  #temp_labelOf = temp_labelOf_gen()
  temp_labelOf = temp_labelOf_gen(d)
  #temp_sameLabel = temp_sameLabel_gen()
  temp_sameLabel = precompute_gen(d,'sameLabel')
  temp_toLeft = precompute_gen(d,'toLeft')
  temp_toRight = precompute_gen(d,'toRight')
  temp_above = precompute_gen(d,'above')
  temp_below = precompute_gen(d,'below')
  temp_within = precompute_gen(d,'within')

  if counting == 'yes':
    ifImg = "(img Img)"
    temp_counts = make_temp_counts(d)
  else:
    ifImg = ""
    temp_counts = ""


  temp = "(or" + ("\n" + temp_neq) + ("\n" + temp_labelOf) + ("\n" + temp_sameLabel) + ("\n" + temp_counts) + ("\n" + temp_toLeft) + ("\n" + temp_toRight) + ("\n" + temp_above) + ("\n" + temp_below) + ("\n" + temp_within) + "\n)"
  result = "(define-fun re ((r Rels)(a Obj)(b Obj)" + ifImg + ") Bool\n" + temp +")"

  preamble = ";Relation evaluator\n" + ";All relations are made into a uniform arity using 'nullobj' as a default argument\n"
  return preamble + result