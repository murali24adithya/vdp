from vdputils import *

def genObjEnumDatatype(problem_dict):
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  objid_list = [objdict['objid'] for img in all_imgs_list for objdict in img]
  res = ''
  for objid in objid_list:
    res = res + '(' + objid + ')' 
  return '(declare-datatypes ((Obj 0))\n((\n{0}\n)))\n\n'.format(res) 


def genObjConsts(problem_dict):
  raw_obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(raw_obj_datatype)
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  objid_list = [objdict['objid'] for img in all_imgs_list for objdict in img]
  res = ''
  for i in range(len(objid_list)):
    res = res + '(define-fun {0} () {1} {2})\n'.format(objid_list[i],obj_datatype,convertConstant(i,raw_obj_datatype))

  return res


def genLabelEnumDatatype(problem_dict):
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  res = ''
  for label in label_set:
    res = res + '(' + label + ')' 
  return '(declare-datatypes ((Label 0))\n((\n{0}\n)))\n\n'.format(res) 


def genLabelConsts(problem_dict):
  raw_label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(raw_label_datatype)
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  res = ''
  for i,elem in enumerate(label_set):
    res = res + '(define-fun {0} () {1} {2})\n'.format(elem,label_datatype,convertConstant(i,raw_label_datatype))

  return res


def genRels(problem_dict,relname):
  #Only supporting object and label datatypes between 'BitVec' and an enumerative datatype at this point. May need to define a different function for the two datatypes.
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  label_datatype = problem_dict['label-datatype']
  label_datatype = makeDatatype(label_datatype)

  if relname == 'labelOf':
    res = ''
    for img in all_imgs_list:
      for objdict in img:
        for label in objdict['labels']:
          res = res + '(and (= obj ' + objdict['objid'] + ') (= lbl ' + label + '))\n'
    return '(define-fun labelOf((obj {0})(lbl {1})) Bool\n(or\n{2}\n))\n\n'.format(obj_datatype,label_datatype,res)
  
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
    return '(define-fun {0} ((x {1})(y {1})) Bool\n(or\n{2}\n))\n\n'.format(relname,obj_datatype,res)

  else:
    raise ValueError('Incorrect relation symbol.')


def makeTempDefs(problem_dict):
  #Temporarily adding definitions for equality/disequality of objects
  ##because these are used in defining flags for formula synthesis
  obj_datatype = problem_dict['obj-datatype']
  obj_datatype = makeDatatype(obj_datatype)
  res = ''
  res = res + '(define-fun eq ((x1 {0})(x2 {0})) Bool (= x1 x2))\n'.format(obj_datatype)
  res = res + '(define-fun neq ((x1 {0})(x2 {0})) Bool (not (= x1 x2)))\n'.format(obj_datatype)
  
  return res



def writeDatatypesAndRels(problem_dict,filename,write_or_append):
  filehandle = open(filename,write_or_append)
  
  #Only supporting object and label datatypes between 'BitVec' and an enumerative datatype at this point.
  if problem_dict['obj-datatype'][0] == 'BitVec':
    filehandle.write(genObjConsts(problem_dict))
  else:
    filehandle.write(genObjEnumDatatype(problem_dict))
  filehandle.write('\n\n')
  
  #Only supporting object and label datatypes between 'BitVec' and an enumerative datatype at this point.
  if problem_dict['label-datatype'][0] == 'BitVec':
    filehandle.write(genLabelConsts(problem_dict))
  else:
    filehandle.write(genLabelEnumDatatype(problem_dict))
  filehandle.write('\n\n')
  
  rels = ['labelOf','leftOf','rightOf','above','below','within']
  for rel in rels:
    filehandle.write(genRels(problem_dict,rel))
  filehandle.write('\n\n')

  #Temporarily adding definitions for equality/disequality of objects
  ##because these are used in defining flags for formula synthesis
  filehandle.write(makeTempDefs(problem_dict))
  filehandle.write('\n\n')

  filehandle.close()