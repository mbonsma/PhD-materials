#!/bin/bash

# run this from the dated folder in which to run BLAST
# inputs: accession, query

# usage:
# bash blast_setup_niagara.sh accession query

# example usage:
# bash blast_setup_niagara.sh SRR1873837 test_query.fasta

# load modules required for blast
module load gcc/7.3.0
module load lmdb/0.9.22
module load gmp/6.1.2
module load boost/1.66.0
module load blast+/2.7.1

# load gnu parallel
module load gnu-parallel/20180322 

cp $HOME/blast_submit_script_niagara.sh . # copy submit script to current folder
cp $HOME/run_blast.sh . # copy blast script to current folder

accession=$1
query=$2

# split query file into a couple of smaller query files - only do this once per query file though
numlines="$(wc -l $query | cut -d' ' -f1)"
newname="$(echo $query | cut -d'.' -f1)"

#if [ $numlines > 400 ]; then
#  split -l 400 $query "$newname"_
#fi

# create serial script for each fasta sub-file

num="$(ls "$accession"/"$accession"* | wc -l)" # get number of files to run
num_query="$(ls "$newname"_* | wc -l)"
ls "$accession"/"$accession"* > accessions.txt
num_runs=$(( $num_query*$num ))
ls "$newname"_* > queries.txt

counter=1
i=$(printf %04d $counter)

# make directory for serialjobdir files
if [ ! -d "$accession"_blast ]; then
  mkdir "$accession"_blast 
fi

while read -r line || [[ -n "$line" ]];
do
  acc="$(echo $line | cut -d"/" -f2)"; # get filename
  echo $acc
  while read -r query || [[ -n "$query" ]];
  do
    echo bash run_blast.sh $acc $query $accession > "$accession"_blast/doserialjob$i.sh; # make run script for each accession
    ((counter+=1));
    i=$(printf %04d $counter);
  done <queries.txt
done <accessions.txt



