#!/bin/bash
# done_sims.sh: get list of completed sims from log file
# works in any dated folder except 2019-05-14
# usage: bash done_sims.sh path_to_date
# example usage: bash done_sims.sh /scratch/g/goyalsid/mbonsma/2019-05-07

date=$1
cd $date # enter dated folder
yes | rm done_sims.txt # remove old done_sims
log=$(ls slurm*.log -t | head -n 1)  # get most recent log file
echo $log
cat $log | sort -n | tail -n +2 > joblog.txt
while read line
do 
  exitval=$(echo $line | cut -d" " -f7)
  folder=$(echo $line | cut -d" " -f10)
  #echo $folder
  #echo $exitval
  if [ $exitval -eq "0" ]; then 
    echo "${folder}" >> done_sims.txt     
  fi
done < joblog.txt
cd -
