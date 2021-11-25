#!/bin/bash

# bash script that runs blast
# usage: bash run_blast.sh accession query datapath

accession=$1
query=$2
datapath=$3

blastn -query "$query" -subject "$datapath"/"$accession" -evalue 0.00001 -dust no -max_target_seqs 10000000 -outfmt 6 -out  "$datapath"_blast/"$accession"_"$query"_blast.txt
