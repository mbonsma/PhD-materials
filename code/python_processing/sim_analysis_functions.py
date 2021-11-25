#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:08:08 2018

@author: madeleine
"""

import numpy as np
import pandas as pd
from scipy import sparse
import os

def recreate_x(x_line):
    """Each line of the data file is a vector x with length 2m + 3
    x[0]: nb0
    x[1:m+1]: nbi
    x[m+1:2m+1]: nvi
    x[2m+2]: C
    x[2m+3]: t
    """
    return np.array(x_line.decode("utf-8").split('[')[1].split(']')[0].split(','), dtype = 'float32')

def recreate_phage(phage_row):
    """
    Input: the list of prophages for a particular timepoint (i.e phage[-1], where phages is read in above)
    Output: the same line formatted as a list of lists of integers
    """
    phage_list = []
    
    phages_string = phage_row.decode("utf-8").split('[')[2:]
    for phage in phages_string:
        phage = phage.split(']')[0]
        phage_list.append(list(np.array(phage.split(','),dtype=int)))
    
    return phage_list
    
def remove_zeros(x,m):
    """
    Remove empty entries in simulation data, i.e. if there are no bacteria or phage at that index
    
    Inputs:
    x: vector of length 2*m + 3 where each entry is a population abundance.
    x[0] = nb0, x[1:m+1] = nbi, x[m+1:2*m+1] = nvi, x[2*m+1] = C, x[2*m + 2] = time  
    m: number of unique phage species in the population
    
    Returns:
    x and m after removing matching zero entries. x is still length 2*m+3, but m may have decreased.
    """
    zero_inds = np.where(np.logical_and(x[m+1:2*m+1] == 0, x[1:m+1] == 0) == True)[0] # where both phage and bac are 0
    if len(zero_inds) == 0:
        return x, m
    else:
        x = np.delete(x, m+1 + zero_inds) # delete phage
        x = np.delete(x, 1 + zero_inds) # delete corresponding bacteria
        m -= len(zero_inds)
        return x, m

def remove_zeros_pop_array(pop_array, max_m, all_phages, phage_size_cutoff = 0):
    """
    Remove empty entries in a block of simulation data, i.e. if there are no bacteria or phage at that index
    during that time range.
    
    Inputs:
    pop_array: array of dimension (time_indices, 2*max_m + 3) (not sparse format)
    pop_array[:,0] = nb0, pop_array[:,1:m+1] = nbi, pop_array[:, m+1:2*m+1] = nvi, 
    pop_array[:,2*m+1] = C, pop_array[:,2*m + 2] = time  
    max_m: number of unique phage species in the population over all time
    phage_size_cutoff: size of phage clone below which to exclude (inclusive) if bacteria is 0 
    
    Returns:
    pop_array, m, and all_phages after removing columns corresponding to phage and bacteria all being zero. 
    """
    
    nbi = pop_array[:,1:max_m+1]
    nvi = pop_array[:,max_m+1:2*max_m+1]
    
    # get nonzero columns
    nonzero = np.logical_or(nvi > phage_size_cutoff, nbi != 0)
    
    # subset nbi and nvi
    new_nbi = nbi[:, np.any(nonzero, axis = 0)]
    new_nvi = nvi[:, np.any(nonzero, axis = 0)]
    
    new_all_phages = np.array(all_phages)[np.any(nonzero, axis = 0)]

    new_max_m = np.sum(np.any(nonzero, axis = 0))
    
    new_pop_array = np.concatenate([pop_array[:,0, np.newaxis], new_nbi, new_nvi, pop_array[:, -2:]], axis = 1)
    
    return new_pop_array, new_max_m, new_all_phages

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest_below(array, value):
    """
    Return the index of the element that is nearest to `value` but lower than `value`
    array must be sorted for this to work
    """
    return(np.argmax((array > value)[1:] > (array > value)[:-1]))
    
def create_all_phages(data, phages):
    """
    Creates a list of all the unique phages that were ever present in the simulation
    
    Inputs: 
    data: file object created from reading "populations.txt" with readlines()
    phages: file object created from reading "protospacers.txt" with readlines()
    
    Output: list of all unique phages in the simulation
    """
    all_phages = []

    for count, row in enumerate(data):
        phage_list = recreate_phage(phages[count]) 

        for ind, phage in enumerate(phage_list): # for each phage sequence, track its abundace
            if phage not in all_phages:
                all_phages.append(phage)
    return all_phages

def create_pop_array(data, all_phages, phages):
    """
    Create array that contains all the simulation data. 
    Rows are time points, columns are unique populations. 
    Each unique population ever present in the simulation has its own column.
    This is not much faster than creating an array directly, 
    but it is much faster to save and load.
    
    pop_array structure:
        
    | Columns                 | Description |
    | 0                       | $n_B^0$     |
    | 1 : max_m + 1`          | $n_B^i$     |
    | max_m + 1 : 2*max_m + 1 | $n_V^i$     |
    | 2*max_m + 2 or -2       | $C$         |
    | 2*max_m + 3 or -1       | $t$ (mins)  |
    
    Inputs:
    data: file object created from reading "populations.txt" with readlines()
    all_phages: list of all unique phages ever present in the simulation
    phages: file object created from reading "protospacers.txt" with readlines()
    
    Outputs:
    pop_array: sparse scipy array of data, structured as above
    max_m: total number of unique species (phage or bacteria) in the simulation
    """
    
    max_m = len(all_phages) # from a previous run
    nrows = len(data) 

    data_vec = []
    i_vec = []
    j_vec = []
    max_j = max_m*2 + 3
    
    for i, row in enumerate(data):
        x = recreate_x(row)
        m = int((len(x) - 3)/2)
        phage_list = recreate_phage(phages[i]) 

        # possibly do i_vec all at once:
        # i_vec += [i]*len(x)

        # nb0
        data_vec.append(x[0])
        i_vec.append(i)
        j_vec.append(0)

        # time
        data_vec.append(x[-1])
        i_vec.append(i)
        j_vec.append(max_j-1)

        # c
        data_vec.append(x[-2])
        i_vec.append(i)
        j_vec.append(max_j-2)

        # add population totals to pop_array
        for count2, phage in enumerate(phage_list):       
            ind = all_phages.index(phage)
            # nbi
            data_vec.append(x[1+count2])
            i_vec.append(i)
            j_vec.append(ind + 1)

            # nvi
            data_vec.append(x[m + 1 + count2])
            i_vec.append(i)
            j_vec.append(max_m + 1 + ind)
            
    pop_array = sparse.coo_matrix((data_vec, (i_vec, j_vec)), shape=(nrows, max_j),  dtype = 'float32')
    pop_array = sparse.csr_matrix(pop_array, dtype = 'float32')
    
    return pop_array, max_m
        



def PV(i,j, pv, e):
    if i == j:
        return pv*(1-e)
    else:
        return pv

def PV_matrix(pv, e, m):
    pv_matrix = np.ones((m,m))
    pv_matrix *= pv
    pv_matrix[np.identity(m, dtype = bool)] = pv*(1-e)
    
    return pv_matrix

def nvdot(nb0, nbi, nvj, C, F, r, g, c0, B, pv, e, eta, alpha):
    """
    nbi and nvj are vectors of length m with the population sizes of spacers and protospacers
    """
    
    nv = np.sum(nvj)
    nbs = np.sum(nbi)
    
    pv_matrix = PV_matrix(pv, e, len(nbi))
    pva_term = np.sum(pv_matrix*np.outer(nbi,nvj))
        
    return -(F + alpha*(nb0 + nbs))*nv + alpha*B*(pv*nb0*nv + pva_term)

def nb0dot(nb0, nbi, nvj, C, F, r, g, c0, B, pv, e, eta, alpha):
    """
    nbi and nvj are vectors of length m with the population sizes of spacers and protospacers
    """
    
    nv = np.sum(nvj)
    nbs = np.sum(nbi)
    
    return (g*C - F)*nb0 - alpha*pv*nb0*nv - alpha*(1-pv)*eta*nb0*nv + r*nbs

def nbsdot(nb0, nbi, nvj, C, F, r, g, c0, B, pv, e, eta, alpha):
    """
    nbi and nvj are vectors of length m with the population sizes of spacers and protospacers
    """
    
    nv = np.sum(nvj)
    nbs = np.sum(nbi)
    
    pv_matrix = PV_matrix(pv, e, len(nbi))
    pva_term = np.sum(pv_matrix*np.outer(nbi,nvj))
    
    return (g*C - F - r)*nbs + alpha*eta*nb0*nv*(1-pv) - alpha*pva_term

def cdot(nb0, nbi, nvj, C, F, r, g, c0, B, pv, e, eta, alpha):
    
    return F*(c0-C) - g*C*(nb0 + np.sum(nbi))

def nbidot(nb0, nbi, nvj, C, F, r, g, c0, B, pv, e, eta, alpha, i):
    """
    nbi and nvj are vectors of length m with the population sizes of spacers and protospacers
    
    Gives the rate of growth for the ith spacer type
    """
        
    pv_vector = np.ones(len(nvj))*pv
    pv_vector[i] = pv*(1-e)
    
    pva_term = np.sum(pv_vector*nbi[i]*nvj)
    
    return (g*C - F - r)*nbi[i] + alpha*eta*nb0*nvj[i]*(1-pv) - alpha*pva_term

def nvjdot(nb0, nbi, nvj, C, F, r, g, c0, B, pv, e, eta, alpha, j):
    """
    nbi and nvj are vectors of length m with the population sizes of spacers and protospacers
    
    Gives the rate of growth for the jth protospacer type
    """
        
    nbs = np.sum(nbi)
    
    pv_vector = np.ones(len(nvj))*pv
    pv_vector[j] = pv*(1-e)
    
    pva_term = np.sum(pv_vector*nvj[j]*nbi)
        
    return -(F + alpha*(nb0 + nbs))*nvj[j] + alpha*B*(pv*nb0*nvj[j] + pva_term)
    
def sum_positions_vec(n_i,max_abund):
    """
    Takes a vector where each position is a spacer type and each value is an abundance
    returns the (non-cumulative) frequency distribution from lowest abundance to highest abundance.
    """
    d = np.zeros((int(max_abund)+1))    
    count = 0
    for i in range(len(n_i)):
        v = int(n_i[i])
        if v != 0:
            d[v] += 1
            count += 1
    # normalize by first value
    #norm_val = d[1]
    #d_normed = []
    #for ind in range(len(d)):
    #    d_normed.append(d[ind]/norm_val)
        
    return d
    
def recreate_parent_list(parent_row):
    """
    Input: list of prophage parents for a particular time point (i.e. parent_list[-1], where parent_list is read
    in above)
    Output: the same line formatted as a list of tuples of integers, or 'nan' if the phage has no parent (i.e. 
    is one of the original phages)
    """
    parent_list_row = []
    parents_string = parent_row.decode("utf-8").split('[')[2:]
    for parent in parents_string:
        parent = parent.split(']')[0]
        if parent == "''": # this is one of the original phages with no back mutations
            parent_list_row.append([])
        else: # has at some point arisen by mutation
            # check if any of the list are blank
            parent = parent.split(',')
            try:
                ind = parent.index("''")
                parent[ind] = np.nan
            except:
                pass
            try:
                ind = parent.index('')
                parent[ind] = np.nan
            except:
                pass
            parent_list_row.append(list(np.array(parent,dtype='float32')))
        
    return parent_list_row

def find_file(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return root, name
        
def load_simulation(folder, timestamp, save_pop = True, return_parents = False):
    #print("loading parameters")
    # load parameters
    parameters = pd.read_csv(folder + "/parameters_%s.txt" %timestamp, delimiter = '\t', header=None)
    parameters.columns = ['parameter', 'value']
    parameters.set_index('parameter')

    f = float(parameters.loc[parameters['parameter'] == 'f']['value'])
    c0 = float(parameters.loc[parameters['parameter'] == 'c0']['value'])
    g = float(parameters.loc[parameters['parameter'] == 'g']['value'])
    B = float(parameters.loc[parameters['parameter'] == 'B']['value'])
    R = float(parameters.loc[parameters['parameter'] == 'R']['value'])
    eta = float(parameters.loc[parameters['parameter'] == 'eta']['value'])
    pv = float(parameters.loc[parameters['parameter'] == 'pv']['value'])
    alpha = float(parameters.loc[parameters['parameter'] == 'alpha']['value'])
    e = float(parameters.loc[parameters['parameter'] == 'e']['value'])
    L = float(parameters.loc[parameters['parameter'] == 'L']['value'])
    mu = float(parameters.loc[parameters['parameter'] == 'mu']['value'])
    m_init = float(parameters.loc[parameters['parameter'] == 'm_init']['value'])
    gen_max = float(parameters.loc[parameters['parameter'] == 'gen_max']['value'])
    max_save = float(parameters.loc[parameters['parameter'] == 'max_save']['value'])
    try:
        theta = float(parameters.loc[parameters['parameter'] == 'theta']['value'])
    except:
        theta = 0.0
        
    # load list of all phages that ever existed
    with open(folder + "/all_phages_%s.txt" %timestamp, "rb") as all_phages_file:
        all_phages = all_phages_file.readlines()
    
    #print("creating list of all phages")
    all_phages = recreate_phage(all_phages[0])
    
    #print("attempting to load pre-made pop_array...")
    try: # try loading pop_array directly
        pop_array = sparse.load_npz(folder + "/pop_array_%s.txt.npz" %timestamp) # fast <3 <3 <3
        max_m = int((pop_array.shape[1] -3)/2)
        #Sprint("loading from existing pop_array file")
            
    except: #if pop_array doesn't exist, load the necessary files and create it
        print("creating pop_array")
        with open(folder + "/populations_%s.txt" %timestamp, "rb") as popdata:
            data = popdata.readlines()
            
        with open(folder + "/protospacers_%s.txt" %timestamp, "rb") as protospacers:
            phages = protospacers.readlines()

        pop_array, max_m = create_pop_array(data, all_phages, phages)
        
        if save_pop == True: # save pop_array so it can be loaded quicker
            #print("saving pop_array")
            sparse.save_npz(folder + "/pop_array_%s.txt" %timestamp, pop_array)
    


    with open("%s/mutation_times_%s.txt" %(folder,timestamp), "rb") as mut_f:
        mutation_t = mut_f.readlines()

    mutation_times = recreate_parent_list(mutation_t[0])

    
    if return_parents == True:
        #print("loading parents and mutation times")
        with open("%s/parents_%s.txt" %(folder,timestamp), "rb") as par:
            parents = par.readlines()
        parent_list = recreate_parent_list(parents[0])
        return f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, max_m, mutation_times, parent_list, all_phages
    
    else:
        return f, c0, g, B, R, eta, pv, alpha, e, L, mu, m_init, gen_max, max_save, theta, pop_array, max_m, mutation_times, all_phages


if __name__ == "__main__":
    pass
    
