#!/bin/bash
# done_sims_2019-05-14.sh: get list of completed sims from log file
# works in the 2019-05-14 dated folder only
# usage: bash done_sims_2019-05-14.sh path_to_date
# example usage: bash done_sims_2019-05-14.sh /scratch/g/goyalsid/mbonsma/2019-05-14

date=$1
cd $date # enter dated folder
yes | rm done_sims.txt # remove old done_sims

while read line
do
  cd $line  # enter run folder
  log=$(ls slurm*.log -t | head -n 1)  # get most recent log file
  echo $log
  cat $log | sort -n | tail -n +2 > joblog.txt # trim header row from log
  while read job
  do 
    exitval=$(echo $job | cut -d" " -f7)
    folder=$(echo $job | cut -d" " -f10) # use the folder name, not the run order
    #echo $exitval
    #echo $folder
    if [ $exitval -eq "0" ]; then
      echo "${line}/${folder}" >> ../done_sims.txt    
    fi
  done < joblog.txt
  cd ..
done < dirs.txt

