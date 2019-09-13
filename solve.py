import os
import time

from problem import *
from smtoutput import *
from utilities import *



def solve(config_dict,problem_dict,smtdict,smt_path):
    start = time.time()
    num_candidates = len(problem_dict['test'].keys())
    resultfile_name = config_dict['folder'] + "smt/results_" + config_dict['instance'] + ".txt"
    for candidate in range(1,num_candidates+1):
        result = "Candidate " + str(candidate) + ":\n" + solve_candidate(candidate,num_candidates,config_dict,problem_dict,smtdict,smt_path)
        result = result + "\n-----------------------------\n"
        resultfile = open(resultfile_name, 'a')
        resultfile.write(result)
        resultfile.close()

    end = time.time()
    timestr = "\n!!!Solved in " + str(end-start) + " seconds."
    resultfile = open(resultfile_name, 'a')
    resultfile.write(timestr)
    resultfile.close()
    print "\nDone."



def solve_candidate(candidate,num_candidates,config_dict,problem_dict,smtdict,smt_path):
    num_rels = config_dict['num_rels']
    num_vars = config_dict['num_vars']
    arity = config_dict['arity']

    assertionstr = assert_candidate_as_answer(candidate,num_candidates)
    countermodels = ""
    result = ""

    num_solutions = config_dict['num_solutions']

    for i in range(1,num_solutions+1):
        #Same files being used for now. Maybe should generate different files for different trials.
        smt_infile_name = config_dict['folder'] + '/smt/' + config_dict['instance'] + '_smtinput.smt2'
        smt_outfile_name = config_dict['folder'] + '/smt/' + config_dict['instance'] + '_smtoutput.txt'

        smt_input = smt_file_gen(problem_dict,smtdict,config_dict,assertionstr+countermodels)
        smt_in_file = open(smt_infile_name, 'w')
        smt_in_file.write(smt_input)
        smt_in_file.close()

        print "Running SMT Query for candidate number " + str(candidate) + ", trial " + str(i)
        smt_call = smt_path + 'z3 ' + smt_infile_name + ' > ' + smt_outfile_name
        os.system(smt_call)
        print "Finished SMT Query."

        (status,smtmodel_dict,pretty_counterformula) = read_single_smt_output(smt_outfile_name,num_rels,num_vars,arity)
        if status == 'unsat':
            result = result + "\nUnsat:No more formulae."
            break
        else:
            countermodels = countermodels + "\n" + counterformula_assertion(smtmodel_dict,config_dict)
            result = result + "\nSolution number " + str(i) + "\n" + pretty_counterformula

    print "*Done with candidate " + str(candidate) + "of " + str(num_candidates) + ".*\n"
    return result



def counterformula_assertion(smtmodel_dict,config_dict):
    result = ""
    num_vars = config_dict['num_vars']
    num_rels = config_dict['num_rels']
    arity = config_dict['arity']

    #for quantifiers
    for i in range(1,num_vars+1):
        qi = "q" + str(i)
        result = result + "(= " + qi + " " + smtmodel_dict[qi] + ")\n"
    #for operators
    for i in range(1,num_rels):
        opi = "op" + str(i)
        result = result + "(= " + opi + " " + smtmodel_dict[opi] + ")\n"
    #for relations, negation operators, and bindings
    for i in range(1,num_rels+1):
        ri = "r" + str(i)
        ni = "n" + str(i)
        result = result + "(= " + ri + " " + smtmodel_dict[ri] + ")\n"
        result = result + "(= " + ni + " " + smtmodel_dict[ni] + ")\n"
        for j in range(1,arity+1):
            argi_j = "arg" + str(i) + "_" + str(j)
            result = result + "(= " + argi_j + " " + smtmodel_dict[argi_j] + ")\n"

    result = "(assert (not (and\n" +  result + ")))"
    return result


# trainfile_name = folder + case+"_train_out.txt"
# testfile_name = folder + case+"_test_out.txt"
# smt_infile_name = folder + instance + '_smtinput.smt2'
# smt_outfile_name = folder + instance + '_smtoutput.txt'
# resultfile_name = folder + 'results_'+ instance +'.txt'
