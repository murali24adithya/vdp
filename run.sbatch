#!/bin/bash
#
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -J vdp_test_run
#SBATCH -p secondary
#SBATCH -o batch_logs.o%j
#SBATCH --gpus=K80:2
##SBATCH -e batch_err.e%j
##SBATCH --mail-user=atharva2@illinois.edu
##SBATCH --mail-type=BEGIN,END
#
# End of embedded SBATCH options
#
python main.py
