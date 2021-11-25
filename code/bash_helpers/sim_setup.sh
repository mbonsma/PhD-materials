#!/bin/bash

# run this from the home folder
# inputs: list of parameters to run, date to create dated folder (must not already exist in $SCRATCH)

# usage:
# bash sim_setup.sh params_list.txt 2018-11-09

mkdir $SCRATCH/$2 # make folder for this submission to run from 
# copy simulation and submission script to SCRATCH
cp simulation_mutating_phage_niagara.py $SCRATCH/$2
cp niagara_submit_script.sh $SCRATCH/$2
cp $1 $SCRATCH/$2

cd $SCRATCH/$2

counter=1
i=$(printf %04d $counter)
#echo $counter

while read -r line || [[ -n "$line" ]];
do
  vars=$line;
  mkdir serialjobdir$i;
  cp simulation_mutating_phage_niagara.py serialjobdir$i; # copy simulation script to each subfolder
  cd serialjobdir$i;
  echo python simulation_mutating_phage_niagara.py $vars > doserialjob$i.sh; # make run script for each set of parameters
  cd ..;
  ((counter+=1));
  i=$(printf %04d $counter);
done <$1
