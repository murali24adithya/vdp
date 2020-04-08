#!/bin/bash

instance=$1
problem_mode=$2

if [ $# -lt 2 ]; then
	echo "Too few arguments. Exiting..."
	exit 2
fi

if [ $problem_mode != "sygus-synth" ] && [ $problem_mode != "smt-check" ] && [ $problem_mode != "smt-synth" ]; then
	echo "Incorrect mode for problem. Specify either sygus-synth, smt-check, or smt-synth."
	exit 2
fi

#Setting file extension depending on problem mode
if [ $problem_mode == 'sygus-synth' ]; then
	extension="sy"
elif [ $problem_mode == 'smt-synth' ] || [ $problem_mode == 'smt-check' ]; then
	extension="smt"
else
	echo "Incorrect problem mode. Specify either sygus-synth or smt-synth"
fi


cd $instance

#In case the argument is the lowermost directory with only one instance
input_file_regex="*_${problem_mode}_file.${extension}"
rm -f $input_file_regex
output_file_regex="*_${problem_mode}_result.txt"
rm -f $output_file_regex
timeoutput_file_regex="*_${problem_mode}_time.txt"
rm -f $timeoutput_file_regex

#In case the argument is an upper level directory
input_files_regex="*/*_${problem_mode}_file.${extension}"
rm -f $input_files_regex
output_files_regex="*/*_${problem_mode}_result.txt"
rm -f $output_files_regex
timeoutput_files_regex="*/*_${problem_mode}_time.txt"
rm -f $timeoutput_files_regex

echo "Finished cleaning up"
echo ""
