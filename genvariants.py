import sys
import os

case = sys.argv[1]
casefolder = "./concepts/" + case +"/"
vfolder = "./variants/"

variantfile = open(casefolder + 'variants.txt','r')
variants = variantfile.read()
variantfile.close()
variants = variants.split("\n")[:20]

for variant in variants:
    if variant == "":
        #empty line
        continue

    [trainlist,testlist] = variant.split("|")
    trainlist = trainlist.split()
    testlist = testlist.split()

    vname = case + "_" + "".join(trainlist) + "-" + "".join(testlist)
    os.system("mkdir -p " + vfolder + vname)
    if len(os.listdir(vfolder+vname)) > 0:
        # variant has already been created
        continue



    for train in trainlist:
        os.system("cp " + (casefolder + train + ".jpg") + " " + (vfolder + vname + "/train_" + train + ".jpg"))
    for test in testlist:
        os.system("cp " + (casefolder + test + ".jpg") + " " + (vfolder + vname + "/test_" + test + ".jpg"))

    os.system("cp " + (casefolder + "config.txt") + " " + (vfolder + vname + "/config.txt"))
