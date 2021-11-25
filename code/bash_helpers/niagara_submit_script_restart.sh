#!/bin/bash
# SLURM submission script for multiple serial jobs on Niagara, resubmitting multiple times
# following this example: https://docs.scinet.utoronto.ca/index.php/FAQ#How_can_I_automatically_resubmit_a_job.3F
# Make sure to change the timeout length to the desired amount (1410m = 23.5 hours) 
#
# Usage: from dated folder directory in $SCRATCH, run the following line, changing the date to the folder name:
# sbatch --export=NUM=0 niagara_submit_script_restart.sh 2019-01-08
# 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00 
#SBATCH --job-name phage-bac-sim

# SET UP THE SIMULATIONS - restart from checkpoint

# copy simulation and submission script to SCRATCH
cd $HOME
cp simulation_mutating_phage_niagara_resume.py $SCRATCH/$1

cd $SCRATCH/$1 # enter dated subfolder

numdir="$(find serialjobdir* -maxdepth 0 | wc -l)"

for counter in `seq 1 $numdir`;
do
  i=$(printf %04d $counter)
  echo $i
  
  if [ -f serialjobdir$i/parameters* ]; then # if the parameters file exists, the simulation has been started and/or completed, so attempt to resume
    cp simulation_mutating_phage_niagara_resume.py serialjobdir$i; # copy simulation script to each subfolder
    cd serialjobdir$i;
    fn="$(ls parameters*)"
    timestamp=${fn:11:26}
    echo python simulation_mutating_phage_niagara_resume.py $timestamp > doserialjob$i.sh; # make run script for each set of parameters
    cd ..;
    echo "simulation started"
  else
    echo "simulation not started, leaving alone"
  fi
  
  chmod u+x serialjobdir$i/doserialjob$i.sh; # make executable

done

# DIRECTORY TO RUN - $SLURM_SUBMIT_DIR is the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR
 
# Turn off implicit threading
export OMP_NUM_THREADS=1

module load NiaEnv/2018a 
module load gnu-parallel/20180322 
module load python/3.6.4-anaconda5.1.0
 
# EXECUTION COMMAND 
timeout --signal=SIGINT 3m parallel --joblog slurm-$SLURM_JOBID.log -j $SLURM_TASKS_PER_NODE "cd serialjobdir{} && ./doserialjob{}.sh" ::: {0001..0086}

# RESUBMIT 10 TIMES HERE
num=$NUM
if [ "$num" -lt 10 ]; then
      num=$(($num+1))
      ssh -t nia-login01 "cd $SLURM_SUBMIT_DIR; sbatch --export=NUM=$num niagara_submit_script_restart.sh $1";
fi
