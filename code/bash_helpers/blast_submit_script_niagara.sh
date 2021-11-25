#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
# Usage: from dated folder directory in $SCRATCH, run the following line, changing the date to the folder name:
# sbatch blast_submit_script.sh SRR1873837
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=04:00:00
#SBATCH --job-name spacer-blast
 
# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR
 
# Turn off implicit threading in Python, R
export OMP_NUM_THREADS=1
 
module load gcc/7.3.0
module load lmdb/0.9.22
module load gmp/6.1.2
module load boost/1.66.0
module load blast+/2.7.1
module load gnu-parallel/20180322 

 
# EXECUTION COMMAND 
parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "bash ./$1_blast/doserialjob{}.sh" ::: {0001..5000}

