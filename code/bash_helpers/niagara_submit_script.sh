#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00 
#SBATCH --job-name phage-bac-sim
 
# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR
 
# Turn off implicit threading
export OMP_NUM_THREADS=1
 
module load gnu-parallel/20180322 
module load python/3.6.4-anaconda5.1.0
 
# EXECUTION COMMAND 
parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "cd serialjobdir{} && ./doserialjob{}.sh" ::: {0001..0450}

