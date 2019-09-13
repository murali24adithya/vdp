# From a dictionary given by the model parser, this function creates a new dictionary of images and the corresponding objects that will be used in the SMT generation
def make_elem(d):
  d_train = d['train']
  d_test = d['test']
  
  result = {}
  training_imgs = d_train.keys()
  for img in training_imgs:
    result[img] = d_train[img].keys()
  
  testing_imgs = d_test.keys()
  for img in testing_imgs:
    result[img] = d_test[img].keys()

  return result

# From dictionary given by the model parser, this functions returns a dictionary of objects and various features like label, top_border, etc that will be used in SMT generation
def make_feature(d,feature):
  if feature == 'label':
    featurefunc = lambda x: x[0]
  elif feature == 'top_border':
    featurefunc = lambda x: x[2]['top']
  elif feature == 'bottom_border':
    featurefunc = lambda x: x[2]['bottom']
  elif feature == 'left_border':
    featurefunc = lambda x: x[2]['left']
  elif feature == 'right_border':
    featurefunc = lambda x: x[2]['right']
  else:
    raise ValueError('Feature not supported.')

  d_train = d['train']
  d_test = d['test']

  result = {}
  training_imgs = d_train.keys()
  for img in training_imgs:
    train_objs = d_train[img].keys()
    for obj in train_objs:
      result[obj] = featurefunc(d_train[img][obj])
  
  testing_imgs = d_test.keys()
  for img in testing_imgs:
    test_objs = d_test[img].keys()
    for obj in test_objs:
      result[obj] = featurefunc(d_test[img][obj])

  return result



def make_labelcount(d):
    result = {}
    d_train = d['train']
    d_test = d['test']

    train_imgs = d_train.keys()
    for img in train_imgs:
        result[img] = {}
        objs = d_train[img].keys()
        lbls = []
        for obj in objs:
            lbls = lbls + d_train[img][obj][0]

        for lbl in set(lbls):
            result[img][lbl] = lbls.count(lbl)

    test_imgs = d_test.keys()
    for img in test_imgs:
        result[img] = {}
        objs = d_test[img].keys()
        lbls = []
        for obj in objs:
            lbls = lbls + d_test[img][obj][0]
        for lbl in set(lbls):
            result[img][lbl] = lbls.count(lbl)

    #print result

    return result






##################
#Bunch of different utility functions for generating various small expressions/strings

def print_config(config_dict):
    result = "Problem instance: " + config_dict['case'] + ","
    result = result + "num_solutions=" + str(config_dict['num_solutions'])
    result = result + ",num_vars=" + str(config_dict['num_vars'])
    result = result + ",num_rels=" + str(config_dict['num_rels'])
    result = result + ",arity=" + str(config_dict['arity'])
    result = result + ",existential=" + config_dict['existential']
    result = result + ",conjuncts=" + config_dict['conjuncts']
    result = result + ",labelconstraint=" + config_dict['labelconstraint']
    result = result + ",counting=" + config_dict['counting']
    print result

def assert_candidate_as_answer(candidate,num_candidates):
    result = ""
    result = result + "(assert (formula_level_1 s" + str(candidate) + "))\n"
    for i in range(1,num_candidates+1):
        if i != candidate:
            result = result + "(assert (not (formula_level_1 s" + str(i) + ")))\n"

    return result



def readconfig(case, config):
    attributes = config.split()
    length = len(attributes)
    if length < 2:
        raise ValueError('Configuration incorrect: ' + config)
    else:
        config_dict = {}
        # mandatory attributes
        config_dict['case'] = case
        config_dict['num_vars'] = int(attributes[0])
        config_dict['num_rels'] = int(attributes[1])
        # default values of optional attributes
        config_dict['arity'] = 2
        config_dict['num_solutions'] = 1
        config_dict['existential'] = 'no'
        config_dict['conjuncts'] = 'no'
        config_dict['labelconstraint'] = 'noneg'
        config_dict['counting'] = 'no'

        for i in range(2, length):
            (knob, value) = attributes[i].split('=')
            if knob == 'arity':
                config_dict[knob] = int(value)
            elif knob == 'num_solutions':
                config_dict[knob] = int(value)
            else:
                config_dict[knob] = value

        instance = case + '_' + str(config_dict['num_vars']) + 'v' + '-' + str(config_dict['num_rels']) + 'r'
        if config_dict['existential'] == 'yes':
            instance = instance + '_' + 'exist'
        if config_dict['conjuncts'] == 'yes':
            instance = instance + '_' + 'and'
        if config_dict['labelconstraint'] != 'no':
            instance = instance + '_' + config_dict['labelconstraint']
        if config_dict['counting'] == 'yes':
            instance = instance + '_' + 'counting'
        config_dict['instance'] = instance

    return config_dict
