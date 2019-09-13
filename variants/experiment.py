import os
import sys

case = sys.argv[1]

os.chdir("../")
os.system("python genvariants.py " + case)
os.chdir("./variants")
folders = sorted(os.listdir("./"))
folders = filter(lambda x: case in x, folders)
length = len(folders)

for i in range(length):
    print "Running folder " + folders[i] + ", which is " + str(i+1) + "of " + str(length) + "."
    os.system("python ../makeinputfiles.py " + folders[i])
    os.system("python ../runyolo.py " + folders[i])
    os.system("timeout 1200 python ../script.py " + folders[i])





