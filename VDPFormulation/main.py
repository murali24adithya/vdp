from processNN import *
from header import *
from sygusSynthDecls import *
from smtCheckDecls import *
from smtSynthDecls import *
from constraints import *

import sys
import math


def generateSygusSynthFile(problem_dict):
  instance = problem_dict['instance']
  synth_filename = instance + '_sygus-synth_file.sy'
  filehandle = open(synth_filename,'w')
  filehandle.write('(set-logic ALL)\n')
  filehandle.write('\n')
  filehandle.close()
  writeDatatypesAndRels(problem_dict,synth_filename,'a')
  writeSygusSynthDecls(problem_dict,synth_filename,'a')
  writeSygusSynthConstraints(problem_dict,synth_filename,'a')
  filehandle = open(synth_filename,'a')
  filehandle.write('\n(check-synth)')
  filehandle.close()


def generateSmtCheckFile(problem_dict):
  instance = problem_dict['instance']
  smt_filename = instance + '_smt-check_file.smt'
  filehandle = open(smt_filename,'w')
  filehandle.write('(set-logic ALL)\n')
  filehandle.write('\n')
  filehandle.close()
  #Datatypes and Rels same as SyGuS file
  writeDatatypesAndRels(problem_dict,smt_filename,'a')
  #New function for SMT Decls. Basically SyGuS decls with empty bodies
  writeSmtCheckDecls(problem_dict,smt_filename,'a')
  #Constraints same as SyGuS but replaced with 'assert' instead of 'constraint'
  writeSmtCheckConstraints(problem_dict,smt_filename,'a')
  filehandle = open(smt_filename,'a')
  filehandle.write('\n(check-sat)')
  filehandle.close()  

def generateSmtSynthFile(problem_dict):
  instance = problem_dict['instance']
  smt_filename = instance + '_smt-synth_file.smt'
  filehandle = open(smt_filename,'w')
  filehandle.write('(set-logic ALL)\n')
  filehandle.write('\n')
  filehandle.close()
  #Datatypes and Rels same as SyGuS file
  writeDatatypesAndRels(problem_dict,smt_filename,'a')
  #New function for SMT based Synth Decls.
  writeSmtSynthDecls(problem_dict,smt_filename,'a')
  #Constraints same as SyGuS but replaced with 'assert' instead of 'constraint'
  #Currently using the guards as a funtion 'gi'. But might later change it to
  #actualy saying 'LabelOf objid synth-label' where the 'synth-label' is the
  #variable being asked to synthesise.
  writeSmtSynthConstraints(problem_dict,smt_filename,'a')
  filehandle = open(smt_filename,'a')
  filehandle.write('\n(check-sat)\n(get-model)')
  filehandle.close()


#instance = 'adoc'
#num_quants = 2
#problem_mode = 'smt-synth'
instance = sys.argv[1]
num_quants = int(sys.argv[2])
problem_mode = sys.argv[3]
problem_dict = genProblemDict(instance,num_quants)
problem_dict['problem-mode'] = problem_mode
if problem_mode == 'sygus-synth':
  #problem_dict['obj-datatype'] = ['BitVec',5]
  #problem_dict['label-datatype'] = ['BitVec',2]
  problem_dict['obj-datatype'] = ['Obj']
  problem_dict['label-datatype'] = ['Label']
  
  generateSygusSynthFile(problem_dict)

elif problem_mode == 'smt-check':
  #problem_dict['obj-datatype'] = ['BitVec',5]
  #problem_dict['label-datatype'] = ['BitVec',2]
  problem_dict['obj-datatype'] = ['Obj']
  problem_dict['label-datatype'] = ['Label']

  problem_dict['quantifier-solution'] = ['' for i in range(num_quants)]
  problem_dict['guard-solution'] = ['' for i in range(num_quants)]
  problem_dict['qf-formula-solution'] = ''
  problem_dict['candidate-solution'] = ''
  
  generateSmtCheckFile(problem_dict)

elif problem_mode == 'smt-synth':
  all_imgs_list = problem_dict['trainimgs'] + problem_dict['candidateimgs']
  objid_list = [objdict['objid'] for img in all_imgs_list for objdict in img]
  res = ''
  label_set = {label for img in all_imgs_list for objdict in img for label in objdict['labels']}
  bitvec_size_obj = math.ceil(math.log(len(objid_list),2))
  bitvec_size_label = math.ceil(math.log(len(label_set),2))
  problem_dict['obj-datatype'] = ['BitVec',bitvec_size_obj]
  problem_dict['label-datatype'] = ['BitVec',bitvec_size_label]
  #problem_dict['obj-datatype'] = ['Obj']
  #problem_dict['label-datatype'] = ['Label']
  generateSmtSynthFile(problem_dict)

else:
  raise ValueError('Specify problem mode for which file to generate.')


#writeDatatypesAndRels(problem_dict,'rels.txt','w')
#writeSynthDecls(problem_dict,'synthdecls.txt','w')
#writeConstraints(problem_dict,'constraints.txt','w')

