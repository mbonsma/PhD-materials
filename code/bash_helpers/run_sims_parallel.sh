# usage: nohup bash run_sims_parallel.sh &
d=$(date +%Y-%m-%d)
export OMP_NUM_THREADS=1
parallel --joblog job$d.log -j 20 "cd serialjobdir{} && bash ./doserialjob{}.sh" ::: {0001..0150}
