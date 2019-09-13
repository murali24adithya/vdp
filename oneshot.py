import sys

# from script import *
# from runyolo import *
# from makeinputfiles import *


case = sys.argv[1]
folder = "./concepts/" + case + "/"
configfile_name = folder + "config.txt"

configfile = open(configfile_name,'r')
configs = configfile.read()
configfile.close()

configs = configs.split("\n")

for config in configs:
    if