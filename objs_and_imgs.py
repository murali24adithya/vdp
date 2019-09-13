# Returns a list of extranous 'objects' that are not real objects but can be used as arguments in relations. Labels are an example.
# This is needed to declare a uniform sort of binding variables that, given a specific valuation, evaluate to objects
def extraneous_objs(d):
  result = {"nullobj"}
  label_lists = d['label'].values()
  for label_list in label_lists:
    result = result | set(label_list)
  return list(result)

# Defines the datatype of images
def declare_datatype_Img(d):
  obj_of_img = d['elem']
  imgs = obj_of_img.keys()
  temp = ""
  for img in imgs:
    temp = temp + " " + img
  result = "(declare-datatypes () ((Img\n" + temp + "\n)))"

  preamble = ";Datatype of images\n"
  return preamble + result

# Defines the datatype of objects
# Extraneous objects that are not real objects also exist
def declare_datatype_Obj(d):
  obj_of_img = d['elem']
  imgs = obj_of_img.keys()
  temp = ""
  for img in imgs:
    for obj in obj_of_img[img]:
      temp = temp + " " + obj
    temp = temp + "\n"
  
  temp = temp + ";extraneous objects\n"
  exobs = extraneous_objs(d)
  for exob in exobs:
    temp = temp + " " + exob 
  result = "(declare-datatypes () ((Obj\n" + temp + "\n)))"

  preamble = ";Constant symbols for objects\n" + ";All values are distinct by default\n"
  return preamble + result

# Defines the predicate that evaluates whether a given object belongs to a given image
def define_fun_inImg(d):
  obj_of_img = d['elem']
  imgs = obj_of_img.keys()
  temp = ""
  for img in imgs:
    for obj in obj_of_img[img]:
      temp = temp + "(and (= x "+ obj +") (= y "+ img +"))\n"
  
  # temp = temp + ";extraneous objects\n"
  # exobs = extraneous_objs(d)
  # for exob in exobs:
  #   for img in imgs:
  #     temp = temp + "(and (= x "+ exob +") (= y "+ img +"))\n"
  
  result = "(define-fun inImg ((x Obj )(y Img)) Bool\n" + "(or\n" + temp + "))"
  
  preamble = ";Membership of objects in images\n" + ";Extraneous objects belong to no images\n"
  return preamble + result