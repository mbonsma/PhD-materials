#!/bin/bash

# run this from the home folder
# this assumes this is a restart of a previous run to continue simulations that didn't finish
# inputs: date for dated folder (MUST ALREADY exist in $SCRATCH)

# usage:
# bash sim_setup_restart.sh 2018-11-09

# copy simulation and submission script to SCRATCH
cp simulation_mutating_phage_niagara_resume.py $SCRATCH/$1
cp niagara_submit_script.sh $SCRATCH/$1

cd $SCRATCH/$1 # enter dated subfolder

numdir="$(find serialjobdir* -maxdepth 0 | wc -l)"

for counter in `seq 1 $numdir`;
do
  i=$(printf %04d $counter)
  
  if [ -f serialjobdir$i/parameters* ]; then # if the parameters file exists, the simulation has been started and/or completed, so attempt to resume
    cp simulation_mutating_phage_niagara_resume.py serialjobdir$i; # copy simulation script to each subfolder
    cd serialjobdir$i;
    fn="$(ls parameters*)"
    timestamp=${fn:11:26}
    echo python simulation_mutating_phage_niagara_resume.py $timestamp > doserialjob$i.sh; # make run script for each set of parameters
    cd ..;
    echo "${i} simulation started"
  else
    echo "${i} simulation not started, leaving alone"
  fi

done
