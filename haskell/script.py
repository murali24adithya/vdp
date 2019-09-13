import sys


def model_parser(model_string):
    model = model_string.split('\n')[1:-1]
    names_and_confidence = model[0::2]
    bounding_boxes = model[1::2]
    parsed_model = ""
    for i in range(len(bounding_boxes)):
        name = "\"" + names_and_confidence[i].split(': ')[0] + "\""
        confidence = "\"" + names_and_confidence[i].split(': ')[1].split('%')[0] + "\""
        boundaries = bounding_boxes[i].split(': ')[1].split(', ')
        left = "\"" + boundaries[0].split('=')[1] + "\""
        right = "\"" + boundaries[1].split('=')[1] + "\""
        top = "\"" + boundaries[2].split('=')[1] + "\""
        bottom = "\"" + boundaries[3].split('=')[1] + "\""
        parsed_model = parsed_model + "(" + name + ","+ confidence + ",[" + left + "," + right + "," + top + "," + bottom+"]),"
    return "[" + parsed_model[:-1] + "]"


def parse_input(string):
    each_model = string.split('Enter')[1:-1]
    all_models = ""
    for i in range(len(each_model)):
        all_models = all_models + model_parser(each_model[i]) + ","
    all_models = "[" + all_models[:-1] + "]"
    return all_models

# print(len(sys.argv))
# print(sys.argv)

readfile = open(sys.argv[1], 'r')
train_str = readfile.read()
train_str = parse_input(train_str)
writefile = open('Trainmodels.hs', 'w')
writefile.write("module Trainmodels where \n\ntrain_models = " + train_str)
readfile.close()
writefile.close()

readfile = open(sys.argv[2], 'r')
test_str = readfile.read()
test_str = parse_input(test_str)
writefile = open('Testmodels.hs', 'w')
writefile.write("module Testmodels where \n\ntest_models = " + test_str)
readfile.close()
writefile.close()
