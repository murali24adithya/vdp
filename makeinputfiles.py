import sys
import os

#def makeinputfiles():

case = sys.argv[1]
pwd = os.getcwd()
folder = pwd + "/" + case + "/"


train_result = ""
test_result = ""
filenames = sorted(os.listdir(folder))
for filename in filenames:
    if ('train' in filename) and ('.jpg' in filename):
        train_result = train_result + (folder + filename) + "\n"
    elif ('test' in filename) and ('.jpg' in filename):
        test_result = test_result + (folder + filename) + "\n"
train_result = train_result[:-1]
test_result = test_result[:-1]

trainfile = open(folder + case+'_train_in.txt','w')
trainfile.write(train_result)
trainfile.close()

testfile = open(folder + case+'_test_in.txt','w')
testfile.write(test_result)
testfile.close()