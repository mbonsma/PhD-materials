#!/bin/bash
# make_tarball.sh: 

# usage: bash make_tarball.sh path_to_date today_date"
# example usage: bash make_tarball.sh /scratch/g/goyalsid/mbonsma/2021-06-11 2021-07-27

date=$1
today=$2
cd $date # enter dated folder
tar -cvzf "${today}_done_sims.tar.gz" -T new_done_sims.txt
cd -
