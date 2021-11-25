#!/bin/bash

# run simulation_stats.py on a list of folders to be analyzed

# usage: 
# bash simulation_stats.sh fileslist
# bash simulation_stats.sh to_analyze.txt

# Description

export OMP_NUM_THREADS=1
fileslist=$1

# iterate through list of folders to add to all_data
while read line
do
  cd $line
  fn="$(ls parameters*)"
  timestamp=${fn:11:26}
  cd -
  echo $timestamp
  python simulation_stats.py $timestamp $line all_params.csv
done < $fileslist
