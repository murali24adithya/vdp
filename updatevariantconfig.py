import sys
import os

case = sys.argv[1]
casefolder = "./concepts/" + case +"/"
vfolder = "./variants/"

variantfile = open(casefolder + 'variants.txt','r')
variants = variantfile.read()
variantfile.close()
variants = variants.split("\n")

for variant in variants:
    if variant == "":
        #empty line
        continue

    variant = "".join(variant.split())
    [trainlist,testlist] = variant.split("|")

    vname = case + "_" + trainlist + "-" + testlist
    os.system("cp " + (casefolder + "config.txt") + " " + (vfolder + vname + "/config.txt"))