#!/bin/bash

# this script runs the steps in Toronto/Research/Code/scinet_notes.txt to download completed sims from a compute canada server.
# this depends on ssh being automatic, i.e. using an ssh password manager 
# inputs: path to supercomputer top folder, dated folder
# run from results folder on hard drive

# usage: 
# bash data_download.sh mbonsma@niagara.scinet.utoronto.ca:/scratch/g/goyalsid/mbonsma 2021-06-11
# bash data_download.sh mbonsma@beluga.computecanada.ca:/scratch/mbonsma 2019-05-07

# 1. Get list of new simulations that haven't been transferred
# 	- get list of completed sims from hard drive
#	- run done_sims script on supercomputer (delete old done_sims.txt first)
# 	- copy done_sims.txt to hard drive
#	- comm -13 completed_sims.txt done_sims.txt > new_done_sims.txt
# 2. Compress new finished simulations on supercomputer
# 	- copy new_done_sims.txt to supercomputer
# 	- compress from list of sims
# 3. Copy compressed archive to hard drive
# 4. Extract compressed archive on hard drive


# 1. Get list of new simulations that haven't been transferred

computerpath=$1
date=$2

cd $date

# get today's date
today=$(printf '%(%Y-%m-%d)T\n' -1)

# get path to ssh to computer
computer=$(echo $computerpath | cut -d":" -f1) 
path=$(echo $computerpath | cut -d":" -f2)

if [[ "$date" == "2019-05-14" ]]; then
  echo "Getting new simulations for $date"
  ## STEP 1
  ls -d run*/serialjobdir* | sort -n > completed_sims.txt
  # get done_sims.txt on server - the script "done_sims_2019-05-14.sh" is in the home folder on server 
  ssh $computer "bash done_sims_2019-05-14.sh $path/$date" 
fi

# for everything except 2019-05-14
if [[ $date != "2019-05-14" ]]; then
  echo "Getting new simulations for $date"
  ## STEP 1
  ls -d serialjobdir* | sort -n > completed_sims.txt
  ssh $computer "bash done_sims.sh $path/$date" # get done_sims.txt on server - the script "done_sims.sh" is in the home folder on server
fi

# for any folder  
scp "$computerpath/$date/done_sims.txt" . # copy done_sims.txt to hard drive
comm -13 completed_sims.txt done_sims.txt > new_done_sims.txt # get new done simulations
len=$(wc -l new_done_sims.txt | cut -d" " -f1) # length of new_done_sims.txt
if [ $len -eq 0 ]; then
  echo "No new simulations to copy"
  exit 0
else
  echo "$len new simulations to copy"
fi

## STEP 2
echo "Copying list of simulations to supercomputer"
scp new_done_sims.txt "$computerpath/$date"
echo "making tar archive"
ssh $computer "bash make_tarball.sh $path/$date $today" # make tar archive
## STEP 3
echo "copying tar archive from supercomputer"
scp "$computerpath/$date/${today}_done_sims.tar.gz" . # copy tar archive to hard drive
## STEP 4
echo "extracting tar archive"
tar -xvzf "${today}_done_sims.tar.gz" # extract tar archive on hard drive
 
# return to upper folder
cd -

