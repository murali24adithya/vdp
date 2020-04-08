#!/bin/bash


pythonscript=/home/umang/Documents/synthesis/VDPFormulation/main.py
#cvc4call=/home/umang/Documents/synthesis/cvc4recent

path_to_instance=$1
numquants=$2
#timeoutbound=${3:-30s}

if [ $# -lt 2 ]; then
	echo "Too few arguments. Exiting..."
	exit 2
fi


if [ ${path_to_instance:(-1)} == "/" ]; then
	path_to_instance=${path_to_instance::(-1)} 
fi

instance=${path_to_instance##*/}   #=> "foo.cpp" (basepath)
instance_dir=${path_to_instance%$instance}  #=> "/path/to/" (dirpath)

#cd ./vdp_smt
cd $instance_dir
cd $instance

python3 $pythonscript $instance $numquants smt-check
echo "Finished generating SMT file for ${instance}"



#time ../../../cvc4recent --lang=sygus2  alldogsonsofas_abc-12d_sygus_file.sy --sygus-abort-size=3 --cegqi-si=none --no-sygus-pbe