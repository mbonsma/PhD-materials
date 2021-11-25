# PhD-materials
A collection of code and notes from my graduate work.

## Running `blast` on the SciNet cluster

The scripts [blast_setup_niagara.sh](https://github.com/mbonsma/PhD-materials/blob/main/code/bash_helpers/blast_setup_niagara.sh), [blast_submit_script_niagara.sh](https://github.com/mbonsma/PhD-materials/blob/main/code/bash_helpers/blast_submit_script_niagara.sh), and [run_blast.sh](https://github.com/mbonsma/PhD-materials/blob/main/code/bash_helpers/run_blast.sh) are for running [blast+](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastDocs&DOC_TYPE=Download) on the [Niagara](https://docs.scinet.utoronto.ca/index.php/Niagara_Quickstart) supercomputer. The `blast+` program is available on Niagara as a module that can be loaded; all you need to do is upload your query and subject sequences. I used this to blast a large number of spacer sequences against metagenomic read data where the enormous fasta file with the reads was split into smaller files of 500 000 reads. This works best if there are either lots of queries or lots of separate subject sub-files so that it can be efficiently parallelized on Niagara; at least 40 serial jobs are needed for best efficiency. 

### Outline of steps:

1. Upload scripts and subject data. `blast_setup_niagara.sh` expects that the subject data is fasta files in a folder called `<accession>` and that each fasta file in the folder is named `<accession_suffix>` (i.e. `SRR1873837` with files `SRR1873837_aa`, etc.). Place `blast_setup_niagara.sh`, subject data, and query fasta file in a folder together.
2. Run `bash blast_setup_niagara.sh accession query` - this creates a series of scripts in a folder called `<accession>_blast` called `doserialjob0001.sh` etc., and copies `blast_submit_script.sh` and `run_blast.sh` into the folder. 
3. Run `sbatch blast_submit_script.sh accession` from the same folder to submit the job to the queue. 

## Simulation data processing
