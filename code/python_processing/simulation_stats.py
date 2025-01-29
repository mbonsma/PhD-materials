#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 14:46:00 2021

@author: madeleine

This script loads simulation files and processes them into an array,
then updates all_data.csv

Requires all_data.csv to be available in the same folder
Requires all_params.csv to be premade and available in the same folder
"""

import numpy as np
import pandas as pd
import re
from scipy import sparse
import argparse

from sim_analysis_functions import (find_nearest, load_simulation)

from spacer_model_plotting_functions import (nbi_steady_state, nvi_steady_state, 
                                             get_trajectories, interpolate_trajectories,
                                             get_clone_sizes, get_large_trajectories, 
                                             fraction_remaining, calculate_speed, 
                                             bac_large_clone_extinction, get_bac_large_trajectories)

def phage_m_to_bac_m(nvi, nb, c0, g, f, alpha, pv, B, n_samples = 15):
    """
    Calculate bacteria m from the distribution of phage clone sizes
    """

    s0 = float(alpha*pv*nb*(B-1) - f*g*c0 - alpha*(1-pv)*nb)
    d0 = float(f*g*c0 + alpha*(1-pv)*nb)

    P0_inf = 1- 2*s0/(B*(s0 + d0)) # extinction probability at long time, independent of nbi
    
    if P0_inf > 1: # can happen if s0 comes out small and negative due to fluctuations in nb
        P0_inf = 1 # set P0 == 1

    N_est = (B*(s0 + d0))/(2*s0) # clone size at which P0 ~ (1/e)
    
    # get list of clone sizes by combining several timepoints
    phage_clone_sizes = (nvi[::int(nvi.shape[0]/n_samples)]).toarray().flatten()
    phage_clone_sizes = np.array(phage_clone_sizes[phage_clone_sizes > 0 ], dtype = 'int')
    
    # list of sizes from 0 to largest observed size
    clone_sizes = np.arange(0, np.max(phage_clone_sizes)+1)

    # survival probability for each clone size
    clone_size_survival = 1 - P0_inf**clone_sizes

    clone_size_survival[int(N_est):] = 1 # set Pest for larger sizes to 1

    # number of clones of size k 
    counts = np.bincount(phage_clone_sizes)

    mean_m = np.sum(clone_size_survival*counts)/n_samples
    
    return mean_m

def simulation_stats(folder, timestamp):
    
    # regex to match a year beginning with 20
    folder_date = re.findall("20[0-9][0-9]-[0-1][0-9]-[0-3][0-9]", folder) 
    
    f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, \
         max_m, mutation_times, all_phages = load_simulation(folder, timestamp);
    
    t_ss = gen_max / 5 # minimun t_ss = 2000, otherwise gen_max/5
        
    #if m_init > 1:
    #    continue

    # check for extinction:
    last_tp = pop_array[-1].toarray().flatten()
    if not np.any(last_tp[:max_m+1] > 0):
        return
    if not np.any(last_tp[max_m+1:2*max_m+1] > 0):
        return
    
    # subsample time if necessary - makes matrix much smaller in cases where Gillespie was heavily used

    # create mask for times that are not near the 0.5 save mark
    # CAUTION: if the save timestep is changed, this will do weird things
    timestep = 0.5
    cutoff = 0.02 # increase the cutoff to keep more points, decrease it to keep fewer
    mask1 = np.ma.masked_inside(pop_array[:, -1].toarray().flatten()*g*c0 % timestep, 0, cutoff).mask 
    new_times = (pop_array[:, -1]*g*c0)[mask1]
    timediffs =  new_times[1:] - new_times[:-1]
    pop_array = pop_array[mask1]

    # create mask for timesteps that are 0 (multi-saving)
    mask2 = ~np.ma.masked_where(timediffs.toarray().flatten() == 0, timediffs.toarray().flatten()).mask
    if type(mask2) != np.bool_: # if nothing is masked, mask2 will be a single value. Only mask if not.
        pop_array = pop_array[1:][mask2]

    #resave as sparse
    pop_array = sparse.coo_matrix(pop_array)
    pop_array = sparse.csr_matrix(pop_array)

    t_ss_ind = find_nearest(pop_array[:,-1].toarray()*g*c0, t_ss)

    if any(x in folder for x in exponential_pv_dates): # then this is a new pv sim
        pv_type = 'exponential'
    elif any(x in folder for x in exponential_pv_025_dates):  # then this is a new pv sim with rate 0.25
        pv_type = 'exponential_025'
    elif any(x in folder for x in theta_pv_dates): # then this is theta function pv
        pv_type = 'theta_function'
    else:
        pv_type = 'binary'
    
    # doing .toarray() is slow and memory-intensive, so do it once per simulation
    nbi = pop_array[t_ss_ind:, 1 : max_m + 1].toarray()
    nvi = pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1].toarray()

    # get trajectories        
    # trim at max size in order to measure establishment rate properly
    time_end = 500 # time in bacterial generations to run trajectories to

    (nvi_trajectories, nbi_trajectories, t_trajectories, nvi_fitness, nbi_fitness, 
         nbi_acquisitions, phage_size_at_acquisition, trajectory_lengths, 
         trajectory_extinct, acquisition_rate, phage_identities) = get_trajectories(pop_array, nvi, nbi, f, 
                                    g, c0, R, eta, alpha, e, pv, B, mu, max_m, m_init, t_ss_ind,
                                    trim_at_max_size = True, aggressive_trim_length = time_end)

    # interpolate trajectories
    fitness_times = np.concatenate([np.arange(0.5,6,0.5), np.arange(6,25,2), 
                                    np.arange(25, 100, 5), np.arange(100, time_end, 10)])
    nvi_interp = interpolate_trajectories(nvi_trajectories, t_trajectories, fitness_times, g, c0)
    
    mean_nvi = np.nanmean(nvi_interp, axis = 1) # conditioned on survival - nan if gone extinct
    mean_phage_fitness = np.gradient(mean_nvi, fitness_times) / mean_nvi
    
    # bacterial spacer acquisition
    nbi_acquisitions = np.sort(np.array(nbi_acquisitions)[~np.isnan(nbi_acquisitions)])
    
    try: 
        t = nbi_acquisitions[int(len(nbi_acquisitions)*0.9)] # time at which 90% of acquisitions have happened
        t_ind = find_nearest(fitness_times, t)
        fitness_at_acquisition = mean_phage_fitness[t_ind]
        mean_ind = find_nearest(fitness_times, np.mean(nbi_acquisitions))
        first_ind = find_nearest(fitness_times, nbi_acquisitions[0])
        
        if t > fitness_times[-1]: # print warning that trajectories aren't long enough
            print(str(timestamp) + " Longer mean trajectories needed: " + str(t) + " > " + str(fitness_times[-1]))
        
        first_acquisition_time = nbi_acquisitions[0]
        median_acquisition_time = nbi_acquisitions[int(len(nbi_acquisitions)/2) - 1]
        fitness_at_mean_acquisition = mean_phage_fitness[mean_ind]
        fitness_at_first_acquisition = mean_phage_fitness[first_ind]
        
        mean_bac_acquisition_time = np.mean(nbi_acquisitions)
        
    except IndexError: # no bacterial acquisitions
        fitness_at_acquisition = np.nan
        first_acquisition_time = np.nan
        median_acquisition_time = np.nan
        fitness_at_mean_acquisition = np.nan
        fitness_at_first_acquisition = np.nan
        mean_bac_acquisition_time = np.nan
        
    # get establishment time

    # calculate predicted large clone extinction
    nv = np.sum(pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1], axis = 1)
    nb = np.sum(pop_array[t_ss_ind:, : max_m+1], axis = 1)
    nb0 = pop_array[t_ss_ind:, 0]
    C = pop_array[t_ss_ind:, -2]

    mean_nb = np.mean(nb[::int(len(nb)/n_snapshots)])
    mean_nv = np.mean(nv[::int(len(nb)/n_snapshots)])
    mean_C = np.mean(C[::int(len(nb)/n_snapshots)])
    mean_nb0 = np.mean(nb0[::int(len(nb)/n_snapshots)])
    
    # get mean field predictions for clone size
    nvi_ss = nvi_steady_state(mean_nb, mean_nv, mean_C, mean_nb0, f, g, c0, e, alpha, B, mu, 
                              pv, R, eta)
    nbi_ss = nbi_steady_state(mean_nb, f, g, c0, e, alpha, B, mu, pv)
    
    # if nvi_ss is negative (happens sometimes)
    while nvi_ss < 0: # recalculate means with different sampling
        shift = np.random.randint(0,100)
        print("negative nvi_ss: %s" %timestamp)
        mean_nb = np.mean(nb[shift::int(len(nb-shift)/n_snapshots)])
        mean_nv = np.mean(nv[shift::int(len(nb-shift)/n_snapshots)])
        mean_C = np.mean(C[shift::int(len(nb-shift)/n_snapshots)])
        mean_nb0 = np.mean(nb0[shift::int(len(nb-shift)/n_snapshots)])
        nvi_ss = nvi_steady_state(mean_nb, mean_nv, mean_C, mean_nb0, f, g, c0, e, alpha, B, mu, 
                              pv, R, eta)
    
    # get phage clone sizes
    (mean_m, mean_phage_m, mean_large_phage_m, mean_large_phage_size, Delta_bac, Delta_phage, 
         mean_nu, e_effective) = get_clone_sizes(pop_array, c0, e, max_m, t_ss_ind, pv_type, theta, all_phages, 1, 
                                                 n_snapshots = n_snapshots)

    # use simulation nbi_ss to get extinction times, same as for nvi    
    bac_extinction_times_large, bac_extinction_times_large_phage_present = bac_large_clone_extinction(pop_array, nbi, nvi,
                                                                        max_m, nbi_ss, t_ss_ind)


    # get large trajectories with size cutoff = nvi_ss
    sim_length_ss = last_tp[-1]*g*c0 - t_ss
    mean_lifetime_large, establishment_rate, establishment_time = get_large_trajectories(nvi_trajectories, 
                    t_trajectories, trajectory_lengths, trajectory_extinct, nvi_ss, g, c0, sim_length_ss)
    
    bac_establishment_rate, establishment_time_bac = get_bac_large_trajectories(nbi_trajectories, 
                                                    t_trajectories, nbi_ss, g, c0, sim_length_ss)

    # get spacer turnover and turnover speed
    turnover_array, interp_times = fraction_remaining(pop_array, t_ss, t_ss_ind, g, c0, gen_max, max_m)
    speed, start_ind = calculate_speed(turnover_array, interp_times)

    F = f*g*c0
    beta = mean_nb*alpha*pv
    delta = F + alpha*mean_nb*(1-pv)
    freq = nvi_ss / mean_nv
    mean_T_backwards_nvi_ss = 2*mean_nv*freq*(1-np.log(freq))*g*c0/((B-1)**2 * beta + delta)
    
    p = beta / (beta + delta)
    predicted_establishment_fraction = (1 - (2-3*B*p + p*B**2)/(B*p*(B-1)))
    
    nvi = pop_array[t_ss_ind:, max_m+1 : 2*max_m + 1]
    rescaled_phage_m = phage_m_to_bac_m(nvi, mean_nb, c0, g, f, alpha, pv, B)
    
    all_mutation_times = []
    
    for times in mutation_times:
        all_mutation_times += list(times)
    
    all_mutation_times = np.sort(all_mutation_times)
    all_mutation_times = all_mutation_times[all_mutation_times > 0]
    all_mutation_times_ss = all_mutation_times[all_mutation_times*g*c0 > t_ss]
    
    mutation_rate_actual = len(all_mutation_times_ss)/((all_mutation_times_ss[-1] - all_mutation_times_ss[0])*g*c0)
    
    # add to data frame
    
    df = pd.DataFrame()
    
    df['C0'] = [c0]
    df['mu'] = [mu]
    df['eta'] = [eta]
    df['e'] = [e]
    df['B'] = [B]
    df['f'] = [f]
    df['pv'] = [pv]
    df['m_init'] = [m_init]
    df['pv_type'] = [pv_type]
    df['gen_max'] = [gen_max]
    df['max_save'] = [max_save]
    df['theta'] = [theta]
    df['t_ss'] = [t_ss]
    df['mean_m'] = [mean_m]
    df['mean_phage_m'] = [mean_phage_m]
    df['mean_large_phage_m'] = [mean_large_phage_m]
    df['mean_large_phage_size'] = [mean_large_phage_size]
    df['rescaled_phage_m'] = [rescaled_phage_m]
    df['timestamp'] = [timestamp]
    df['folder_date'] = folder_date
    df['mean_nu'] = [mean_nu]
    df['mean_nb'] =  [mean_nb]
    df['mean_nv'] = [mean_nv]
    df['mean_C'] = [mean_C]
    df['Delta_bac'] = [Delta_bac]
    df['Delta_phage'] = [Delta_phage]
    df['e_effective'] = [e_effective]
    df['fitness_discrepancy'] = [mean_phage_fitness[0]]
    df['mean_size_at_acquisition'] = [np.nanmean(phage_size_at_acquisition)] # mean phage clone size at the time that a spacer is acquired, ignoring trajectories for which no spacer is acquired
    df['std_size_at_acquisition'] = [np.nanstd(phage_size_at_acquisition)]# std dev phage clone size at the time that a spacer is acquired, ignoring trajectories for which no spacer is acquired
    df['fitness_at_90percent_acquisition'] = [fitness_at_acquisition]
    df['fitness_at_mean_acquisition'] = [fitness_at_mean_acquisition]
    df['fitness_at_first_acquisition'] = [fitness_at_first_acquisition]
    df['num_bac_acquisitions'] = [len(nbi_acquisitions)]
    df['mean_bac_acquisition_time'] = [mean_bac_acquisition_time]
    df['median_bac_acquisition_time'] = [median_acquisition_time]
    df['first_bac_acquisition_time'] = [first_acquisition_time]
    df['mean_large_trajectory_length_nvi_ss'] = [mean_lifetime_large] 
    df['mean_trajectory_length'] = [np.mean(trajectory_lengths)]
    df['mean_T_backwards_nvi_ss'] =  [mean_T_backwards_nvi_ss]
    df['mean_bac_extinction_time'] = [np.mean(bac_extinction_times_large)*g*c0] # simulation average
    df['mean_bac_extinction_time_phage_present'] = [np.mean(bac_extinction_times_large_phage_present)*g*c0]
    df['establishment_rate_nvi_ss'] = [establishment_rate]
    df['turnover_speed'] = [speed]
    df['predicted_establishment_fraction'] = [predicted_establishment_fraction]
    df['measured_mutation_rate'] = [mutation_rate_actual]
    df['mean_establishment_time'] = [establishment_time]
    df['max_m'] = [max_m]
    df['establishment_rate_bac'] = [bac_establishment_rate]
    df['mean_bac_establishment_time'] = [establishment_time_bac]
    
    # add mean_m to dataframe by joining on parameters that vary
    new_data = all_params.merge(df, on = ['C0', 'mu', 'eta', 'e', 'B', 'f', 'pv' 'm_init', 'theta'])
    
    try:
        all_data.columns == new_data.columns
    except:
        raise
        
    # add new data to original df
    result = pd.concat([all_data, new_data], sort = True).reset_index()
    result = result.drop("index", axis = 1)
    
    result = result.drop_duplicates()
    
    result.to_csv("all_data.csv")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Simulation analysis')
    parser.add_argument('timestamp',type=str,help='timestamp to analyze')
    parser.add_argument('folder',type=str,help='folder')
    parser.add_argument('all_params',type=str,default='all_params.csv',
                        nargs='?',help='filename for parameters csv')
    
    args = parser.parse_args()
    
    
    # define parameters
    timestamp = args.timestamp
    folder = args.folder
    all_params_fn = args.all_params
    
    all_params = pd.read_csv(all_params_fn, index_col=0)
    all_data = pd.read_csv("all_data.csv", index_col=0)
    
    n_snapshots = 50 # number of points to sample (evenly) to get population averages

    exponential_pv_dates = ["2019-06-24", "2021-09-09"]
    exponential_pv_025_dates = ["2021-02-01", "2021-09-08"]
    theta_pv_dates = ["2021-06-11", "2021-08-26", "2021-09-13"]

    if np.sum(all_data['timestamp'].isin([timestamp])) == 0: # then this timestamp has not been analyzed
        simulation_stats(folder, timestamp)
