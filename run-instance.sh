#!/bin/bash

#This script takes in a path and executes 'synth.py' with the right configuration file 
##(right now default configuration is num_quants=2 and full grammar)
#'synth.py' takes in the name of an instance, configurations and generates the sygus file called <instance> + '_sygus_file.sy'
#'synth.py' resides in the variable called  synthpath. This script goes inside the folder, generates the sygus file with 'synth.py' and the configurations,
##runs the sygus file with a time command and a timeout (and possibly an argument for how many solutions we want from it),
##and records the results in a file called <instance> + '_sygus_results.txt'


#####TO DO:
#Consider adding support for running the python file and the generated sygus file with multiple configurations.
scriptpath="/home/umang/Documents/synthesis/VDPFormulation"
cvc4call="/home/umang/Documents/synthesis/cvc4recent"

instance=$1
numquants=$2
problem_mode=$3
timeoutbound=${4:-30s}

if [ $# -lt 3 ]; then
	echo "Too few arguments. Exiting..."
	exit 2
fi

#Setting lang argument and file extension for cvc4 call
if [ $problem_mode == "sygus-synth" ]; then
	lang="sygus2"
	extension="sy"
elif [ $problem_mode == "smt-synth" ]; then
	lang="smt"
	extension="smt"
	#Need to set produce-models option if we want to get satisfying assignments
	cvc4call="${cvc4call} --produce-models"
else
	echo "Incorrect problem mode. Specify either sygus-synth or smt-synth"
fi


cd ./vdp_smt
cd $instance

for d in *; do
	#for every variation in the concept instance
	echo "Processing" $d
	cd $d
	python3 "${scriptpath}/main.py" $d $numquants $problem_mode
	inputfile="${d}_${problem_mode}_file.${extension}"
	outputfile="${d}_${problem_mode}_result.txt"
	timeoutputfile="${d}_${problem_mode}_time.txt"
	
	if [ "$timeoutbound" == "no-timeout" ]; then
		(time $cvc4call --lang=$lang $inputfile > $outputfile) 2>&1 | tee $timeoutputfile;
	else
		(timeout $timeoutbound time $cvc4call --lang=$lang $inputfile; echo "$d" > $outputfile;\
	 if [ $? -eq 124 ] ; then echo "${d} Timed out" ; else echo "$d" ; fi) 2>&1 | tee $timeoutputfile
	fi
	
	#Prettyprint the solution for the smt-synth case
	if [ $problem_mode == "smt-synth" ]; then
		tempoutfile="${d}_${problem_mode}_tempresult.txt"
		(python3 "${scriptpath}/prettySmtSynthSolution.py" $outputfile; echo "" ) 2>&1 | tee $tempoutfile
		cat $outputfile >> $tempoutfile
		rm $outputfile
		mv $tempoutfile $outputfile
	fi

	echo "Finished processing" $d
	echo ""
	cd ../
done


#timeout 30s time $cvc4call --lang=sygus2 $sygusfile > $sygusoutput
#time ../../../cvc4recent --lang=sygus2  alldogsonsofas_abc-12d_sygus_file.sy --sygus-abort-size=3 --cegqi-si=none --no-sygus-pbe