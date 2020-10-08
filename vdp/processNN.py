# def processNNOutputOneImage(raw_img,imgid,img_type):
#   if img_type == 't':
#     img_tag = 't'
#   elif img_type == 'c':
#     img_tag = 'c'
#   else:
#     raise ValueError('Incorrect image type: must be either \'t\' or \'c\' ')

#   s = raw_img.split('\n')[1:-1]
#   img_list = []
#   i = 0
#   while i < len(s):
#     raw_label = s[i]
#     raw_bounding_box = s[i+1]
    
#     objdict = {}
#     label = raw_label.split(':')[0]
#     left_margin = raw_bounding_box.split(':')[1].split(',')[0].split('=')[1]
#     top_margin = raw_bounding_box.split(':')[1].split(',')[1].split('=')[1]
#     right_margin = raw_bounding_box.split(':')[1].split(',')[2].split('=')[1]
#     bottom_margin = raw_bounding_box.split(':')[1].split(',')[3].split('=')[1]

#     objdict['label'] = label
#     objdict['left'] = int(left_margin)
#     objdict['top'] = int(top_margin)
#     objdict['right'] = int(right_margin)
#     objdict['bottom'] = int(bottom_margin)
#     objdict['objid'] = img_tag + str(imgid) + 'o' + str(i//2 + 1)
#     objdict['imgid'] = img_type + str(imgid)
#     img_list = img_list + [objdict]

#     i = i+2

#   return img_list


def processNNOutputOneImage(raw_img,imgid,img_type):
  if img_type == 't':
    img_tag = 't'
  elif img_type == 'c':
    img_tag = 'c'
  else:
    raise ValueError('Incorrect image type: must be either \'t\' or \'c\' ')
  
  
  attributes = raw_img.split('\n')[1:-1]
  #Image representation will be a list of object dictionaries
  img = []

  running_objid = 1
  objdict = {'labels':[]}
  for attribute in attributes:
    components = attribute.split(': ')
    if components[0] != 'Bounding Box':
      #Has to be one of the labels of the object
      label = components[0]
      #Transform label to handle multi-word labels like 'teddy bear' instead of 'cat'
      label = '_'.join(label.split(' '))
      objdict['labels'] = objdict['labels'] + [label]

      #Not incorporating confidence percentages in the object representation at this point
      confidence = int(components[1][:-1])

    else:
      #Bounding box. If there are multiple ones, the first one will be chosen
      #Support has to be better for this. The logic could go wrong if there
      ##are boxes without labels, labels without boxes, etc. Solution is to
      ## modify YOLO to give more structured output
      boundaries = components[1]
      left_margin = boundaries.split(', ')[0].split('=')[1]
      top_margin = boundaries.split(', ')[1].split('=')[1]
      right_margin = boundaries.split(', ')[2].split('=')[1]
      bottom_margin = boundaries.split(', ')[3].split('=')[1]
      
      objdict['left'] = int(left_margin)
      objdict['top'] = int(top_margin)
      objdict['right'] = int(right_margin)
      objdict['bottom'] = int(bottom_margin)

      #Construcut other attributes of the object and add the dictionary to the list
      objdict['objid'] = img_tag + str(imgid) + 'o' + str(running_objid)
      objdict['imgid'] = img_tag + str(imgid)
      img = img + [objdict]
      
      #Update running object id counter and refresh object dictionary
      running_objid = running_objid + 1
      objdict = {'labels':[]}

  return img


def processNNOutputManyImages(name,img_type):
  filehandle = open(name,'r')
  s = filehandle.read()
  s = s.split('Enter')[1:-1]
  list_of_imgs = []
  for i in range(len(s)):
    list_of_imgs = list_of_imgs + [processNNOutputOneImage(s[i],i+1,img_type)]
  filehandle.close()
  return list_of_imgs


def genProblemDict(instance,num_quants):
  trainfile_name = instance + '_train_out.txt'
  candidatefile_name = instance + '_test_out.txt'
  train_imgs_list = processNNOutputManyImages(trainfile_name,'t')
  candidate_imgs_list = processNNOutputManyImages(candidatefile_name,'c')
  #all_imgs_list = train_imgs_list + candidate_imgs_list
  
  vars_list = ['x'+str(i+1) for i in range(num_quants)]
  formula = ['f'] + vars_list
  
  problem_dict = {}
  problem_dict['instance'] = instance
  problem_dict['formula'] = formula
  problem_dict['num-quants'] = num_quants
  problem_dict['trainimgs'] = train_imgs_list
  problem_dict['candidateimgs'] = candidate_imgs_list
  problem_dict['grammar-flag'] = ''

  return problem_dict