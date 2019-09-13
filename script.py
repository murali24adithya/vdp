import sys
import os
import time
#import argparse

from io import *
from solve import *
from utilities import *

# parser =argparse.ArgumentParser()
#
# #name of problem instance
# parser.add_argument("problem", help="This is the problem instance. You should have a folder inside ./concepts of this name containing the training and candidate images.")
# #number of variables
# parser.add_argument("num_vars",help="This is the number of variables i.e., the number of quantifiers in the desired formula.",type=int)
# #num of relations
# parser.add_argument("num_rels",help="This is the number of atomic relations that will appear in the desired formula. It is a proxy for length of formula.", type=int)
#
# #whether the formula should contain only existential quantification
# parser.add_argument("-ex", "--exists", help="Should only existentials be used? 'yes' or 'no', default='no')
#
# args = parser.parse_args()
# print args

smt_path = '../z3/bin/'


case = sys.argv[1]

start = time.time()


folder = "./" + case + "/"
configfile = open(folder + "config.txt",'r')
configs = configfile.read()
configfile.close()

os.system('mkdir -p ' + folder + 'smt')
os.system('rm ' + folder + 'smt/*.*')

trainfile_name = folder + case + "_train_out.txt"
testfile_name = folder + case + "_test_out.txt"
(problem_dict,smtdict) = input_files(trainfile_name,testfile_name)
# #print problem_dict['test']

configs = configs.split('\n')

for config in configs:
    if config == '':
        #empty line
        continue

    config = config.split("\n")[0]
    config_dict = readconfig(case,config) #Also creates the 'instance' attribute
    config_dict['folder'] = folder
    print_config(config_dict)

    #solve(num_solutions,problem_dict,smtdict,num_vars,num_rels,arity,existential,conjuncts,labelconstraint,counting,smt_path)
    solve(config_dict,problem_dict,smtdict,smt_path)

end = time.time()
timestr = "The entire config file took " + str(end-start) + " seconds."
timefile = open(folder + 'configtime.txt','w')
timefile.write(timestr)
timefile.close()




# # print sys.argv
# num_vars = int(sys.argv[2])
# num_rels = int(sys.argv[3])
# arity = 2
# existential = sys.argv[4]
# conjuncts = sys.argv[5]
# labelconstraint = sys.argv[6]
# case = sys.argv[1]
# folder = "./concepts/" + case + "/"


# instance = case + '_' + str(num_vars) + 'v' + '-' + str(num_rels) + 'r'
# if existential == 'yes':
#     instance = instance + '_' + 'exist'
# if conjuncts == 'yes':
#     instance = instance + '_' + 'and'
# if labelconstraint != 'no':
#     instance = instance + '_' + labelconstraint
#
# trainfile_name = folder + case+"_train_out.txt"
# testfile_name = folder + case+"_test_out.txt"
# smt_infile_name = folder + instance + '_smtinput.smt2'
# smt_outfile_name = folder + instance + '_smtoutput.txt'
# resultfile_name = folder + 'results_'+ instance +'.txt'
#
# #(problem_dict,smtdict) = input_files(sys.argv[1],sys.argv[2])
# (problem_dict,smtdict) = input_files(trainfile_name,testfile_name)
# #print problem_dict['test']
# smt_input = smt_file_gen(problem_dict,smtdict,num_vars,num_rels,arity,existential,conjuncts,labelconstraint)
#
# smt_in_file = open(smt_infile_name,'w')
# smt_in_file.write(smt_input)
# smt_in_file.close()
#
# print 'SMT query generated. Querying..'
#
# smt_call = '../z3/bin/z3 ' + smt_infile_name + ' > ' + smt_outfile_name
# os.system(smt_call)
#
# print 'SMT query returned. Writing results ..'
#
# results  = read_smt_output(smt_outfile_name,num_rels,num_vars,arity)
# resultfile = open(resultfile_name,'w')
# resultfile.write(results)
# resultfile.close()
#
# print 'Done.'
