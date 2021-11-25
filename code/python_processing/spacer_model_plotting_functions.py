# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:46:16 2017

@author: madeleine
"""

import numpy as np
import collections
from scipy.interpolate import interp1d
from scipy.integrate import solve_bvp
from scipy.integrate import solve_ivp
from sim_analysis_functions import load_simulation, find_nearest

def running_mean(x, N):
    """
    From https://stackoverflow.com/a/27681394
    """
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def zfun(p,B,pv0,f):
    """
    Steady state solution for z without CRISPR
    """
    return p*(B-1/pv0)/(1+p*(B-1/pv0))

def yfun(p,B,pv0,f):
    """
    Steady state solution for y without CRISPR
    """
    return (B*p-p/pv0-f*(B*p-p/pv0+1))/(p*(B*p-p/pv0+1))

def xfun(p,B,pv0,f):
    """
    Steady state solution for x without CRISPR
    """
    return f/(B*p-p/pv0)

def xi_sigmoid(xiH,xiL,xR,n,x):
    """
    returns 1-xi = e as a function of x
    """
    return xiL + (xiH-xiL)*((x**n)/(x**n+xR**n))

def table_to_contour(vals_array):
    """
    Input: n x 3 array with columns 1 and 2 x-y coordinates and column 3 contour height
    Output: xvals, yvals, zvals, formatted for the 'contour' function 
    """
    xvals = vals_array[:,0] # first column
    yvals = vals_array[:,1] # second column
    zvals = vals_array[:,2] # third column

    xdim = len(np.unique(xvals)) # number of unique xcoords
    ydim = len(np.unique(yvals)) # number of unique ycoords

    xvals = np.reshape(xvals,(xdim,ydim))
    yvals = np.reshape(yvals,(xdim,ydim))
    zvals = np.reshape(zvals,(xdim,ydim))
    
    return xvals, yvals, zvals 
    
def lineslice(vals_array,param_slice):
    """
    vals_array is in rows of (xi, param, variable)
    
    param_slice is the value on the y axis that you want to take a slice at. 
    It can be approximate, this function will find the closest point to it.
    """
    xaxis = np.unique(vals_array[:,0]) # take the unique xi coordinates
    idx = np.argmin(np.abs(vals_array[:,1]-param_slice))
    param = vals_array[idx,1]
    inds = np.where(vals_array[:,1]==param)
    plotvals = vals_array[inds,2][0]
    
    return xaxis, plotvals, param

def lineslice_vert(vals_array,param_slice):
    """
    vals_array is in rows of (xi, param, variable)
    
    param_slice is the value of xi on the x axis that you want to take a slice at. 
    It can be approximate, this function will find the closest point to it.
    """
    idx = np.argmin(np.abs(vals_array[:,0]-param_slice))
    param = vals_array[idx,0]
    inds = np.where(vals_array[:,0]==param)
    plotvals = vals_array[inds,2][0]
    yaxis = vals_array[inds,1][0]
    
    return yaxis, plotvals, param
    
def cubsol3(a,b,c,d):
    """
    One of three roots of a generic cubic equation:
    a*x^3 + b*x^2 + c*x + d == 0
    Note: I force the positive root of the square root so that it's continuous
    """
    square_root = np.sqrt(4*(-b**2 + 3*a*c)**3 + (-2*b**3 + 9*a*b*c - 
    27*a**2*d)**2)
    square_root = np.real(square_root) + (1j)*np.abs(np.imag(square_root)) # take the positive imaginary part of root
    
    return -b/(3.*a) + ((complex(1,-np.sqrt(3)))*(-b**2 + 3*a*c))/(3.*2**0.6666666666666666*a*(-2*b**3 + 
    9*a*b*c - 27*a**2*d + square_root)**0.3333333333333333) - ((complex(1,np.sqrt(3)))*(-2*b**3 + 9*a*b*c - 
    27*a**2*d + square_root)**0.3333333333333333)/(6.*2**0.3333333333333333*a)

def cubsol2(a,b,c,d):
    return -b/(3.*a) + ((complex(1,np.sqrt(3)))*(-b**2 + 3*a*c))/(3.*2**0.6666666666666666*a*(-2*b**3 + 
    9*a*b*c - 27*a**2*d + np.sqrt(4*(-b**2 + 3*a*c)**3 + (-2*b**3 + 9*a*b*c - 
    27*a**2*d)**2))**0.3333333333333333) - ((complex(1,-np.sqrt(3)))*(-2*b**3 + 9*a*b*c - 
    27*a**2*d + np.sqrt(4*(-b**2 + 3*a*c)**3 + (-2*b**3 + 9*a*b*c - 
    27*a**2*d)**2))**0.3333333333333333)/(6.*2**0.3333333333333333*a)

def cubsol1(a,b,c,d):
    return -b/(3.*a) - (2**0.3333333333333333*(-b**2 + 3*a*c))/(3.*a*(-2*b**3 + 
    9*a*b*c - 27*a**2*d + np.sqrt(4*(-b**2 + 3*a*c)**3 + (-2*b**3 + 9*a*b*c - 
    27*a**2*d)**2))**0.3333333333333333) + (-2*b**3 + 9*a*b*c - 
    27*a**2*d + np.sqrt(4*(-b**2 + 3*a*c)**3 + (-2*b**3 + 9*a*b*c - 
    27*a**2*d)**2))**0.3333333333333333/(3.*2**0.3333333333333333*a)
    
def aterm(f,p,pv0,e,B,R,eta):
    """
    For the cubic equation for nu in the four-variable model with CRISPR,
    this is the coefficient of nu^3
    """
    return B*e**2*f*p*pv0**2*(-1 + f + R)

def bterm(f,p,pv0,e,B,R,eta):
    """
    For the cubic equation for nu in the four-variable model with CRISPR,
    this is the coefficient of nu^2
    """
    return -(e*f*pv0*(pv0*(f + R) + p*(1 + f*(-1 + B*(-eta + (1 + e + eta)*pv0)) - 
                                       R + B*(eta + pv0*(-1 - e - eta + 2*R)))))

def cterm(f,p,pv0,e,B,R,eta):
    """
    For the cubic equation for nu in the four-variable model with CRISPR,
    this is the coefficient of nu
    """
    return f*(pv0*(f*(-eta + pv0 - (1 - e)*pv0 + eta*pv0) + pv0*R) + p*(eta*(-1 + f) - 
        (1 + 2*B)*eta*(-1 + f)*pv0 + (1 - e)*(1 + B*eta)*(-1 + f)*pv0 - pv0*(-1 + f + R) + 
        B*pv0**2*(-1 - 2*eta - (1 - e)*(1 + eta)*(-1 + f) + f + 2*eta*f + R)))

def dterm(f,p,pv0,e,B,R,eta):
    """
    For the cubic equation for nu in the four-variable model with CRISPR,
    this is the coefficient of nu^0
    """
    if type(R) != float and type(R) != int and type(R) != np.float64:
        return np.ones(len(R))*(-(eta*f*(-1 + pv0)*(f*pv0 + (-1 + f)*p*(-1 + B*pv0))))
    elif type(e) != float and type(e) != int and type(e) != np.float64:
        return np.ones(len(e))*(-(eta*f*(-1 + pv0)*(f*pv0 + (-1 + f)*p*(-1 + B*pv0))))
    else:
        return (-(eta*f*(-1 + pv0)*(f*pv0 + (-1 + f)*p*(-1 + B*pv0))))

def x_fn_nu(nu, f, p, pv0, e, B, R, eta):
    """
    Steady-state solution for x as a fn of nu, 
    where nu is calculated at steady-state using cubsol3
    """
    return f/(B*(1 - e*nu)*p - p/pv0)

def z_fn_nu(nu, f, p, pv0, e, B, R, eta):
    """
    Steady-state solution for z as a fn of nu, 
    where nu is calculated at steady-state using cubsol3
    """
    return (B*(1 - e*nu)*p - p/pv0)/(1 + B*(1 - e*nu)*p - p/pv0)

def y_fn_nu(nu, f, p, pv0, e, B, R, eta):
    """
    Steady-state solution for y as a fn of nu, 
    where nu is calculated at steady-state using cubsol3
    """
    return (B*(1 - e*nu)*p - f*(1 + B*(1 - e*nu)*p - p/pv0) - 
            p/pv0)/((1 - e*nu)*p*(1 + B*(1 - e*nu)*p - p/pv0))

def nbi_steady_state(mean_nb, f, g, c0, e, alpha, B, mu, pv):
    """
    Calculate deterministic mean bacterial clone size at steady state
    """
    L = 30
    P0 = np.exp(-mu*L)
    F = f*g*c0
    return (1/e)*(mean_nb - (F+alpha*mean_nb)/(alpha*B*P0*pv))
    
def nvi_steady_state(mean_nb, mean_nv, mean_C, mean_nb0, f, g, c0, e, alpha, B, mu, pv, R, eta):
    """
    Calculate deterministic mean phage clone size at steady state
    """
    F = f*g*c0
    r = R*g*c0
    nbi_ss = nbi_steady_state(mean_nb, f, g, c0, e, alpha, B, mu, pv)
    
    nvi_ss = nbi_ss*(alpha*pv*mean_nv - (g*mean_C - F - r))/(alpha*eta*mean_nb0*(1-pv) + alpha*pv*e*nbi_ss)
    return nvi_ss

def analytic_steady_state(pv, e_effective, B, R, eta, f, c0, g, alpha):
    # e_effective is a number that gives the average spacer effectiveness for the population. 
    # e_effective can be calculated directly from simulation, or it can be approximated as e/m
    
    p = pv*alpha/g
    flist = np.array([f-0.001, f, f+0.001])

    a = aterm(flist,p,pv,e_effective,B,R,eta).astype(complex)
    b = bterm(flist,p,pv,e_effective,B,R,eta).astype(complex)
    c = cterm(flist,p,pv,e_effective,B,R,eta).astype(complex)
    d = dterm(flist,p,pv,e_effective,B,R,eta).astype(complex)

    nustar = cubsol3(a, b, c, d)
    nustar = np.real(nustar)
    xstar = x_fn_nu(nustar, flist, p, pv, e_effective, B, R, eta)
    ystar = y_fn_nu(nustar, flist, p, pv, e_effective, B, R, eta)
    zstar = z_fn_nu(nustar, flist, p, pv, e_effective, B, R, eta)

    f_ind = np.where(flist == f)
    Nv = ystar[f_ind]*c0
    
    # return Nb, Nv, C, and nu predicted from mean field
    return (xstar[f_ind]*c0, Nv, zstar[f_ind]*c0, nustar[f_ind])

def nvi_growth(nbi, nb, f, g, c0, alpha, e, pv, B, mu, normed=True):
    """
    Single clone growth rate for phages with mutations.
    Does not take into account mutants that enter that phage type (low rate).
    Inputs:
    nbi : 1D vector in time of corresponding bacteria clone sizes
    nb : 1D vector in time of total bacteria population size
    f, g, c0, alpha, e, pv, B, mu : simulation parameters

    Output: 
    s : 1D vector of phage clone growth rate per bacterial generation (not multiplied by population size)
    """

    L = 30 # protospacer length
    P0 = np.exp(-mu*L)
    delta = f*g*c0 + alpha*nb*(1-pv) + alpha*pv*e*nbi
    beta = alpha*pv*nb - alpha*pv*e*nbi

    if normed == True: # normalize to phage death rate
        s = (beta*(B*P0-1) - delta) / delta 
    elif normed == False:
        s = (beta*(B*P0-1) - delta)
        
    return s/(g*c0) # per bacterial generation


def nbi_growth(nbi, nvi, nb, nb0, nv, C, f, R, g, c0, alpha, e, pv, eta):
    """
    Single clone growth rate for bacteria.
    Inputs:
    nbi : 1D vector in time of bacteria clone sizes
    nvi : 1D vector in time of corresponding phage clone sizes
    nb : 1D vector in time of total bacteria population size
    nb0 : 1D vector in time of number of bacteria without spacers
    nv : 1D vector in time of total phage population size
    C : 1D vector in time of total nutrients
    f, g, c0, alpha, e, pv, eta : simulation parameters

    Output: 
    s : 1D vector of phage clone growth rate per bacterial generation (not multiplied by population size)
    """
    F = f*g*c0
    r = R*g*c0

    s = g*C - F - r - alpha*pv*(nv - e*nvi) + alpha*eta*nb0*nvi*(1-pv)/nbi
        
    return s/(g*c0) # per bacterial generation

# ------------- mean field equations with phage mutations -----------

def nvdot(nv, nb0, nbs, e_effective, F, alpha, B, pv):
    """
    Calculates the rate of change in minutes for the total phage population, nv.
    Note that this assumes CRISPR immunity is only present for exactly matching clones.
    
    Inputs:
    nv : total phage population size at time t
    nb0 : number of bacteria without spacers at time t
    nbs : number of bacteria with spacers at time t
    e_effective: average overlap between bac and phage populations
    F : chemostat flow rate
    alpha : phage adsorption rate
    B : phage burst size
    pv : probability of phage infection success without CRISPR
    """
    pva = pv*(1-e_effective)
    return -(F + alpha*(nb0+nbs))*nv + alpha*B*pv*nb0*nv + alpha*B*nbs*nv*pva


def nbdot(nv, nb0, nbs, C, e_effective, F, g, alpha, pv):
    """
    Calculates the rate of change in minutes for the total bacterial population.
    
    Inputs:
    nv : total phage population size at time t
    nb0 : number of bacteria without spacers at time t
    nbs : number of bacteria with spacers at time t
    C : nutrient concentration at time t
    e_effective: average overlap between bac and phage populations
    F : chemostat flow rate
    g : bacterial growth rate
    alpha : phage adsorption rate
    pv : probability of phage infection success without CRISPR
    """
    nb = nbs + nb0
    pva = pv*(1-e_effective)
    return (g*C - F)*nb - alpha*nv*(pva*nbs + pv*nb0)

def nbsdot(nv, nb0, nbs, C, e_effective, F, g, alpha, pv, r, eta):
    """
    Calculates the rate of change in minutes for the number of bacteria with spacers.
    
    Inputs:
    nv : total phage population size at time t
    nb0 : number of bacteria without spacers at time t
    nbs : number of bacteria with spacers at time t
    C : nutrient concentration at time t
    e_effective: average overlap between bac and phage populations
    F : chemostat flow rate
    g : bacterial growth rate
    alpha : phage adsorption rate
    pv : probability of phage infection success without CRISPR
    r : rate of spacer loss
    eta : probability of bacterial spacer acquisition
    """
    
    pva = pv*(1-e_effective)
    return (g*C - F - r)*nbs + alpha*(1-pv)*eta*nb0*nv - alpha*nv*pva*nbs

def nb0dot(nv, nb0, nbs, C, e_effective, F, g, alpha, pv, r, eta):
    """
    Calculates the rate of change in minutes for the number of bacteria with spacers.
    
    Inputs:
    nv : total phage population size at time t
    nb0 : number of bacteria without spacers at time t
    nbs : number of bacteria with spacers at time t
    C : nutrient concentration at time t
    e_effective: average overlap between bac and phage populations
    F : chemostat flow rate
    g : bacterial growth rate
    alpha : phage adsorption rate
    pv : probability of phage infection success without CRISPR
    r : rate of spacer loss
    eta : probability of bacterial spacer acquisition
    """
    
    return (g*C - F)*nb0 + r*nbs - alpha*pv*nb0*nv - alpha*(1-pv)*eta*nb0*nv

def cdot(C, nb, F, c0, g):
    """
    Inputs:
    C : nutrient concentration at time t
    nb : total bacteria population size at time t
    F : chemostat flow rate
    c0 : initial nutrient concentration
    g : bacterial growth rate
    """
    return F*(c0-C) - g*C*nb

def nvidot(nv, nb, nvi, nbi, F, alpha, B, pv, e, mu):
    """
    Calculates the rate of change in minutes for an individual phage clone.
    This ignores mutations into this phage type (low rate).
    
    Inputs:
    nv : total phage population size at time t
    nb : total bacteria population size at time t
    nvi : abundance of a single phage clone (not a vector)
    nbi : abundance of corresponding bacteria clone (not a vector)
    F : chemostat flow rate
    alpha : phage adsorption rate
    B : phage burst size
    pv : probability of phage infection success without CRISPR
    e : relative immunity provided by CRISPR (0 to 1)
    mu : phage mutation rate
    """
    L = 30 # protospacer length
    P0 = np.exp(-mu*L)
    return -(F + alpha*nb)*nvi + alpha*B*P0*pv*nvi*(nb - e*nbi)


def nbidot(nv, C, nb0, nvi, nbi, F, g, alpha, pv, e, r, eta):
    """
    Calculates the rate of change in minutes for an individual phage clone.
    This ignores mutations into this phage type (low rate).
    
    Inputs:
    nv : total phage population size at time t
    C : nutrient concentration at time t
    nb0 : number of bacteria without spacers at time t
    nvi : abundance of a single phage clone (not a vector)
    nbi : abundance of corresponding bacteria clone (not a vector)
    F : chemostat flow rate
    g : bacterial growth rate
    alpha : phage adsorption rate
    pv : probability of phage infection success without CRISPR
    e : relative immunity provided by CRISPR (0 to 1)
    r : rate of spacer loss
    eta : probability of bacterial spacer acquisition
    """

    return (g*C - F - r)*nbi - alpha*pv*nbi*(nv - e*nvi) + alpha*eta*nb0*nvi*(1 - pv)

def timestep(nb0,nbs,nv,nb,C,nvi,nbi,dt, e_effective, F, g, c0, alpha, pv, r, eta, B, mu, e):
    nb = nb0 + nbs
    nb0_new = nb0 + nb0dot(nv, nb0, nbs, C, e_effective, F, g, alpha, pv, r, eta)*dt
    nbs_new = nbs + nbsdot(nv, nb0, nbs, C, e_effective, F, g, alpha, pv, r, eta)*dt
    nv_new = nv + nvdot(nv, nb0, nbs, e_effective, F, alpha, B, pv)*dt
    c_new = C + cdot(C, nb, F, c0, g)*dt
    nvi_new = nvi + nvidot(nv, nb, nvi, nbi, F, alpha, B, pv, e, mu)*dt
    nbi_new = nbi + nbidot(nv, C, nb0, nvi, nbi, F, g, alpha, pv, e, r, eta)*dt

    return (nb0_new, nbs_new, nv_new, c_new, nvi_new, nbi_new)

def nvi_nbi_ode(t, y, mean_nb, mean_nv, mean_C, mean_nb0, f, g, c0, e, alpha, B, mu, pv, R, eta):
    """
    Calculates the rate of change in minutes for an individual phage clone.
    This ignores mutations into this phage type (low rate).
    
    Inputs:
    t : time to solve at
    y : y[0] = nvi, y[1] = nbi
    """
    L = 30 # protospacer length
    F = f*g*c0
    r = R*g*c0
    P0 = np.exp(-mu*L)
    return [-(F + alpha*mean_nb)*y[0] + alpha*B*P0*pv*y[0]*(mean_nb - e*y[1]), 
            (g*mean_C - F - r)*y[1] - alpha*pv*y[1]*(mean_nv - e*y[0]) + alpha*eta*mean_nb0*y[0]*(1 - pv)]

def z_ode(t, y, p, B):
    """
    Solving the regular P0 equation using the ODE solver
    
    t : time to solve at
    y : y[0] = zeta(t)
    """

    return 1 - p + p*y[0]**B - y[0]

def zeta_nbi_nvi_ode(t, y, nb, C, nv, nb0, nbi_ss, f, g, c0, alpha, B, pv, e, R, eta, mu, nbi_norm = True):
    """
    Solving the regular P0 equation using the ODE solver (changing s > 0)
    
    t : time to solve at, in minutes
    y : y[0] = nvi, y[1] = nbi, y[2] = zeta(t)

    
    """
    F = f*g*c0
    r = R*g*c0

    beta = alpha*pv*nb
    delta = F + alpha*nb*(1-pv)

    L = 30 # protospacer length
    P0 = np.exp(-mu*L)
    
    dnvi = (-(F + alpha*nb)*y[0] + alpha*B*P0*pv*y[0]*(nb - e*y[1]))
    dnbi = ((g*C - F - r)*y[1] - alpha*pv*y[1]*(nv - e*y[0]) + alpha*eta*nb0*y[0]*(1 - pv))
    
    if nbi_norm == True:
        # nbi normalized by p_0
        if y[1] / (1 - y[2]) > nbi_ss:
            nbi_val = nbi_ss
        else:
            nbi_val = y[1] / (1-y[2])
    
    if nbi_norm == False:
        # nbi not normalized by p_0, capped at nbi_ss
        if y[1]  > nbi_ss:
            nbi_val = nbi_ss
        else:
            nbi_val = y[1]
    
    # straight deterministic nbi prediction
    #nbi_val = y[1]
    
    s = (beta - delta - 2*alpha*pv*e*nbi_val) / (delta + alpha*pv*e*nbi_val)
    
    dzeta = (beta + delta)*(1/(s + 2) + y[2]**B * (s + 1)/(s + 2) - y[2])
     
    return dnvi, dnbi, dzeta

def numerical_nvi_nbi(mean_nb, mean_nv, mean_nb0, mean_C, nvi_ss, f, g, c0, alpha, B, pv, e, mu, R, eta, 
                      dt = 0.1, n_iter = 50000000, epsilon = 10**-1):
    """
    Inputs:
    mean_nb : mean steady state bacteria population (either from sim or e/m)
    mean_nv : mean steady state phage population (either from sim or e/m)
    mean_C : mean steady state nutrient concentration 
    mean_nb0 : mean steady state number of bacteria without spacers

    F, g, c0, alpha, B, pv, e, mu, r, eta : simulation parameters
    dt : numerical solution timestep
    n_iter : number of iterations for numerical solution

    Returns:
    timevec : timecourse for numerical solution (bacterial generations)
    nvi_vec : numerical mean phage clone size over time
    nbi_vec : numerical mean bacteria clone size over time
    """
    
    F = f*g*c0
    r = R*g*c0
    
    # initial clone sizes
    nvi = 1
    nbi = 0

    nvi_vec = [nvi]
    nbi_vec = [nbi]
    timevec = [0]
    
    for i in range(1, n_iter):
        nvi_new = nvi + nvidot(mean_nv, mean_nb, nvi, nbi, F, alpha, B, pv, e, mu)*dt
        nbi_new = nbi + nbidot(mean_nv, mean_C, mean_nb0, nvi, nbi, F, g, alpha, pv, e, r, eta)*dt
        if i%500 == 0:
            nvi_vec.append(nvi_new)
            nbi_vec.append(nbi_new)
            timevec.append(dt*(i+1)*g*c0)
            if i > n_iter / 100: # make sure it runs for at least that long
                if i%10000 == 0:   
                    if np.abs(1 - (np.mean(nvi_vec[-500:]) / nvi_ss)) < epsilon:
                        break
        
        nvi = nvi_new
        nbi = nbi_new
        
    return timevec, nvi_vec, nbi_vec

def nbi_t_analytic(n0, nv, C, nb0, nvi_ss, g, c0, f, R, alpha, pv, e, eta, times):
    
    """
    solve nbi equation analytically for 1D assumption that nvi is nvi_ss
    n0 : initial clone size
    times : in bacterial generations
    """
    
    F = f*g*c0
    r = R*g*c0
    a = g*C - F - r - alpha*pv*(nv-e*nvi_ss)
    b = alpha*eta*nb0*nvi_ss*(1-pv)
    
    return n0*np.exp(a*(times/(g*c0))) + (b/a)*(np.exp(a*(times/(g*c0))) - 1)

def nvi_t_analytic(nb, g, c0, f, alpha, pv, mu, B, L, times):
    
    """
    solve nvi equation analytically for 1D assumption that nbi = 0
    times : in bacterial generations
    """
    
    F = f*g*c0
    growth_rate = alpha*B*pv*np.exp(-mu*L)*nb - F - alpha*nb
    
    return np.exp(growth_rate*(times/(g*c0)))

def get_trajectories(pop_array, nvi, nbi, f, g, c0, R, eta, alpha, e, pv, B, mu, max_m, m_init, t_ss_ind, 
                     remove_bac_nonzero = True, split_small_and_large = False, size_cutoff = 1, trim_at_max_size = False, aggressive_trim = False,
                     aggressive_trim_length = 50, return_fitness = True):
    """
    Get individual clone trajectories and statistics for an individual simulation.
    
    Inputs: 
    pop_array : pop_array in sparse format
    f : chemostat flow rate = F/(g c0)
    g : bacterial growth rate
    c0 : incoming nutrient concentration
    R : rate of spacer loss = r/(g c0)
    eta : bacterial spacer acquisition probability
    alpha : rate of phage encounter with bacteria
    e : probability of matching spacer effectively blocking phage
    pv : probability of successful phage infection in absence of spacer
    B : phage burst size
    mu : phage mutation rate
    max_m : total number of unique protospacers in simulation
    m_init : initial number of protospacers in simulation
    t_ss_ind : index of when t > t_ss generations
    remove_bac_nonzero : if True, any trajectories that have non-zero corresponding bacterial population will be 
        skipped. If False, all phage trajectories will be included.
    trim_at_max_size : if True, trajectories will be saved from their beginning until they reach their max size.
        nbi trajectories will still match nvi trajectories in length. If False, entire trajectories from 
        beginning to extinction will be saved.
    aggressive_trim : if True, trajectories will be trimmed to a max length of aggressive_trim_length. If False,
        entire trajectories will be saved.       
    aggressive_trim_length : max trajectory length in bacterial generations to save if aggressive_trim == True
    
    Returns:
    nvi_trajectories : a list of arrays, where each array is a phage clone trajectory from 0 until it goes extinct or the simulation ends.
    nbi_trajectories : a list of arrays, where each array is a bacteria clone trajectory that matches the corresponding phage
                        clone trajectory in nvi_trajectories
    t_trajectories : a list of arrays, where each array is the time vector in minutes associated with  
                        each trajectory in nvi_trajectories. (Note: not normalized to trajectory start time.)
    nvi_fitness : a list of arrays, where each array is the growth rate of phage corresponding to nvi_trajectories.
    nbi_fitness : a list of arrays, where each array is the growth rate of bacteria corresponding to nbi_trajectories.
    nbi_acquisitions : a vector the same length as nvi_trajectories with either nan (if no spacer is acquired) or a
                        time in bacterial generations relative to the start of the phage trajectory at which a 
                        spacer is acquired.
    phage_size_at_acquisition : a vector the same length as nvi_trajectories with the phage clone size at the time
                        that a spacer is acquired (or nan if no spacer is acquired).

    trajectory_lengths : a vector the same length as nvi_trajectories with the total time in bacterial generations 
                        before a trajectory goes extinct. If the end of the simulation is reached before extinction, 
                        this number is the trajectory length until the simulation ends.
    trajectory_extinct : a vector the same length as nvi_trajectories with True if a trajectory goes extinct, 
                        and False if the end of the simulation is reached before extinction.
    acquisition_rate : mean number of bacteria spacer acquisition events per bacterial generation
    phage_identities : index in pop_array of which phage the trajectory corresponds to
    """

    nvi_trajectories = []
    nbi_trajectories = []
    nbi_acquisitions = []
    phage_size_at_acquisition = []
    t_trajectories = []
    trajectory_lengths = []
    trajectory_lengths_small = []
    trajectory_lengths_large = []
    trajectory_extinct = []
    nvi_fitness = []
    nbi_fitness = []
    phage_identities = []

    nvi_nonzero = pop_array[t_ss_ind:, max_m + 1: 2*max_m+1] != 0
    # this new version keeps it sparse longer
    # captures transitions between False and True i.e. where nvi goes from 0 to >0
    rows, cols = np.where((nvi_nonzero[1:] > nvi_nonzero[:-1]).toarray()) 

    nb = np.array(np.sum(pop_array[t_ss_ind:, : max_m+1], axis = 1)).flatten()
    nv = np.array(np.sum(pop_array[t_ss_ind:, max_m+1 : 2*max_m+1], axis = 1)).flatten()
    nb0 = pop_array[t_ss_ind:, 0].toarray()
    nbs = np.array(np.sum(pop_array[t_ss_ind:, 1: max_m+1], axis = 1)).flatten()
    C = pop_array[t_ss_ind:, -2].toarray()
    t_all = pop_array[t_ss_ind:, -1].toarray().flatten()

    for i in range(len(cols)):
        nvi_traj = nvi[rows[i]:, cols[i]]
        phage_identities.append([cols[i]]) # index of pop_array where this trajectory comes from

        try: # split if multiple extinctions occur
            death_ind = np.where(nvi_traj == 0)[0][1] +1
            nbi_traj = nbi[rows[i] : rows[i] + death_ind, cols[i]]
            nb_traj = nb[rows[i] : rows[i] + death_ind]
            nb0_traj = nb0[rows[i] : rows[i] + death_ind]
            nv_traj = nv[rows[i] : rows[i] + death_ind]
            nbs_traj = nbs[rows[i] : rows[i] + death_ind]
            C_traj = C[rows[i] : rows[i] + death_ind]
            t_i = t_all[rows[i] : rows[i] + death_ind]
            nvi_traj = nvi_traj[:death_ind]
            traj_extinct_bool = True
        except:
            nbi_traj = nbi[rows[i] :, cols[i]]
            nb_traj = nb[rows[i] :]
            nb0_traj = nb0[rows[i] :]
            nv_traj = nv[rows[i] : ]
            nbs_traj = nbs[rows[i] :]
            C_traj = C[rows[i] : ]
            t_i = t_all[rows[i] :]
            traj_extinct_bool = False

        if remove_bac_nonzero == True: # continue if trajectory has a non-zero starting bacteria population
            if nbi_traj[0] != 0:
                continue

        trajectory_extinct.append(traj_extinct_bool)
        # get times at which nbi acquires spacer
        try:
            acquisition_time = (t_i - t_i[0])[np.where(nbi_traj > 0)[0][0]]
            nbi_acquisitions.append(acquisition_time*g*c0)
            phage_size_at_acquisition.append(nvi_traj[np.where(nbi_traj > 0)[0][0]])
        except:
            nbi_acquisitions.append(np.nan)
            phage_size_at_acquisition.append(np.nan)

        # save trajectory info
        trajectory_lengths.append((t_i[-1] - t_i[0])*g*c0) # get trajectory total length
        
        if split_small_and_large == True: # return trajectory lengths for small and large trajectories separately
            if np.any(nvi_traj > size_cutoff):
                trajectory_lengths_large.append(t_i[-1] - t_i[np.where(nvi_traj >= size_cutoff)[0][0]])
            else:
                trajectory_lengths_small.append(t_i[-1] - t_i[0])
        
        # get index for aggressive_trim in time
        aggressive_trim_ind = find_nearest((t_i - t_i[0])*g*c0, aggressive_trim_length)
                                            
        if trim_at_max_size == True: # chop off at max size or aggressive_trim length, whichever is later
            trim_ind_max = np.argmax(nvi_traj)+2
            trim_ind = np.max([aggressive_trim_ind, trim_ind_max])

        elif aggressive_trim == True: # keep only early part of trajectories
            trim_ind = np.min([len(nvi_traj), aggressive_trim_ind])

        else: # keep entire trajectory
            trim_ind = -1
            
        t_trajectories.append(t_i[:trim_ind])
        nvi_trajectories.append(nvi_traj[:trim_ind]) 
        nbi_trajectories.append(nbi_traj[:trim_ind])
        if return_fitness == True:
            nvi_fitness.append(nvi_growth(nbi_traj[:trim_ind], nb_traj[:trim_ind], f, g, c0, 
                                          alpha, e, pv, B, mu, normed = False))
            nbi_fitness.append(nbi_growth(nbi_traj[:trim_ind], nvi_traj[:trim_ind], nb_traj[:trim_ind], 
                                          nb0_traj[:trim_ind], nv_traj[:trim_ind], 
                                          C_traj[:trim_ind], f, R, g, c0, alpha, e, pv, eta))
        
    # make predicted acquisition timecourse, accounting for double-acquisitions
    acquisition_rate = (alpha*eta*np.mean(nb0)*(1-pv)*np.mean(nv))/(g*c0)
    #total_acquisitions = np.sum(~np.isnan(nbi_acquisitions))
    #predicted_nbi_acquisitions = (1/acquisition_rate)*total_acquisitions/np.arange(total_acquisitions, 0 , -1)

    if return_fitness == True:
        return (nvi_trajectories, nbi_trajectories, t_trajectories, nvi_fitness, nbi_fitness, nbi_acquisitions, phage_size_at_acquisition,  
            trajectory_lengths, trajectory_extinct, acquisition_rate, phage_identities)
    elif split_small_and_large == True:
        return (nvi_trajectories, nbi_trajectories, t_trajectories, nbi_acquisitions, phage_size_at_acquisition,  
            trajectory_lengths, trajectory_lengths_small, trajectory_lengths_large, trajectory_extinct, acquisition_rate, phage_identities)
    else:
        return (nvi_trajectories, nbi_trajectories, t_trajectories, nbi_acquisitions, phage_size_at_acquisition,  
            trajectory_lengths, trajectory_extinct, acquisition_rate, phage_identities)

def interpolate_trajectories(pop_trajectories, t_trajectories, new_times, g, c0):
    """
    Get fitness for individual clone trajectories from an individual simulation
    
    Inputs: 
    pop_trajectories : a list of trajectories, output of get_trajectories
    t_trajectories : a list of the time axis of trajectories in pop_trajectories, 
        output of get_trajectories
    new_times : timeseries to interpolate over (in bacterial generations)
    g : bacterial growth rate
    
    Returns:
    trajectory_interp : an array with interpolated trajectories as each column where the rows correspond to the values of new_times.
                    To get the mean across trajectories, use np.nanmean(trajectory_interp, axis = 1)
    """
    
    #new_times = np.concatenate([np.arange(0.5,30,0.5), np.arange(30,100,5), np.arange(100,1000,50)])
    trajectory_interp = np.zeros((len(new_times), len(pop_trajectories)))

    # loop through trajectories
    for i, traj in enumerate(pop_trajectories):
        t_i = t_trajectories[i]
        t_traj = (t_i - t_i[0])*g*c0

        trajectory_interp[:,i] = np.interp(new_times, t_traj,
                                         traj, right = np.nan)

    return trajectory_interp
  
def get_large_trajectories(nvi_trajectories, t_trajectories, trajectory_lengths, trajectory_extinct, 
                           size_cutoff, g, c0, sim_length):
    """
    Calculate the mean trajectory length for phages clones exceeding size_cutoff
    
    Inputs:
    nvi_trajectories : list of partial phage trajectories returned by get_trajectories
    t_trajectories : list of times in minutes associated with nvi_trajectories, returned by get_trajectories
    trajectory_lengths : list of total lengths in bacterial generations for trajectories in nvi_trajectories,
                        returned by get_trajectories
    trajectory_extinct : list of True or False, True if trajectory goes extinct, False if trajectory survives 
                        until simulation end
    size_cutoff : phage clone size cutoff (units of phage number) returned by get_clone_sizes
    g : bacterial growth rate
    c0 : incoming nutrient concentration
    sim_length : the length of the steady-state portion of the simulation in bacterial generations 
    
    Returns:
    mean_lifetime_large : mean trajectory length in bacterial generations for trajectories starting at size size_cutoff
    establishment_rate : the average number of phage establishments per bacterial generation
    mean_establishment_time : the mean time in bacterial generations from clone appearance to reaching the size cutoff
    """
    
    t_trajectories_endpoints = []
    t_establish = []
    for i, nvi in enumerate(nvi_trajectories):
        if np.any(nvi > size_cutoff):
            t_i = t_trajectories[i]
            trajectory_large = t_i[np.where(nvi >= size_cutoff)[0][0]:]
            t_establish.append(trajectory_large[0] - t_i[0]) # respective to trajectory start
            
            if trajectory_extinct[i] == True: # include only if trajectory goes extinct
                large_trajectory_length = trajectory_lengths[i] - (trajectory_large[0] - t_i[0])*g*c0
                t_trajectories_endpoints.append(large_trajectory_length)
    
    t_establish = np.array(t_establish)*g*c0
    
    # establishments per bacterial generation
    try:
        #establishment_rate = len(t_establish) / (t_establish[-1] - t_establish[0] )  
        establishment_rate = len(t_establish) / sim_length # denominator should actually be total steady-state sim length
    except IndexError:
        establishment_rate = 0

    mean_lifetime_large = np.mean(t_trajectories_endpoints)
    mean_establishment_time = np.mean(t_establish)
    
    return mean_lifetime_large, establishment_rate, mean_establishment_time

def get_bac_large_trajectories(nbi_trajectories, t_trajectories, size_cutoff, g, c0,
                               sim_length):
    """
    Get the establishment time for bacterial trajectories relative to phage mutation
    
    Inputs:
    nbi_trajectories : list of partial bacteria trajectories returned by get_trajectories
    t_trajectories : list of times in minutes associated with nbi_trajectories, returned by get_trajectories
    size_cutoff : bacteria clone size cutoff (units of bacteria number) 
    g : bacterial growth rate
    c0 : incoming nutrient concentration
    sim_length : the length of the steady-state portion of the simulation in bacterial generations 

    Returns:
    establishment_rate : rate of establishment for bacterial clones
    mean_establishment_time : mean time in bacterial generations that bacteria
        clones establish relative to phage mutation

    """
    t_establish = []
    for i, nbi in enumerate(nbi_trajectories):
        if np.any(nbi > size_cutoff):
            t_i = t_trajectories[i]
            trajectory_large = t_i[np.where(nbi >= size_cutoff)[0][0]:]
            t_establish.append(trajectory_large[0] - t_i[0]) # respective to trajectory start
            
    t_establish = np.array(t_establish)*g*c0
    
    # establishments per bacterial generation
    try:
        #establishment_rate = len(t_establish) / (t_establish[-1] - t_establish[0] )  
        establishment_rate = len(t_establish) / sim_length # denominator should actually be total steady-state sim length
    except IndexError:
        establishment_rate = 0
        
    mean_establishment_time = np.mean(t_establish)
    
    return establishment_rate, mean_establishment_time
    
def PV(i,j, pv, e, pv_type, all_phages):
    """
    Probability of success for phage j against bacterium i
    
    For binary pv, pv(i,j) = pv(1-e) if i=j, pv otherwise.
    
    For exponential pv, pv(i,j) falls off exponentially as the phage gets more
    mutations instead of being binary.
    If the distance is 0, PV = pv(1-e),
    if the distance is very large, PV ~ pv. 
    
    Inputs:
    i : index of bacteria type in nbi list (relative to start of nbs)
    j : index of phage type in nvi list (relative to start of phages)
    e : spacer effectiveness parameter
    pv_type : string that determines which pv function is applied
    all_phages : list of binary sequences for all phages in pop_array columns
    
    """
    distance = np.sum(np.abs(np.array(all_phages[i]) - np.array(all_phages[j])))
    
    if pv_type == 'binary':
        if i == j:
            return pv*(1-e)
        else:
            return pv
    elif pv_type == 'exponential':
        return pv*(1 -  e * np.exp(-distance))
    elif pv_type == 'exponential_025':
        return pv*(1 -  e * np.exp(-0.25* distance))

def fraction_remaining(pop_array, t_ss, t_ss_ind, g, c0, gen_max, max_m, shift=5):
    """
    Calculate the fraction of bacterial spacer types remaining at time t
    as a function of the time delay (interp_times).
    
    Inputs:
    pop_array : pop_array in sparse format
    t_ss : time in bacterial generations at which simulation is assumed to be 
        in steady-state
    t_ss_ind : index corresponding to t_ss along the time axis of pop_array
    g : parameter g
    c0 : parameter c0
    gen_max : maximum bacterial generations for the simulation
    max_m : maximum protospacers in the simulation
    shift : time spacing in bacterial generations for interpolation
        
    Returns:
    turnover_array : square array of dimension len(interp_times) with each row
        being the fraction of bacterial spacers remaining at each time delay. 
        Increasing row number is a later starting point for the calculation.
        Time delays after the simulation endpoint are padded with np.nan.
    interp_times - t_ss : time shift axis in bacterial generations 
    """
    
    
    timepoints = pop_array[t_ss_ind-1:,-1].toarray()*g*c0

    interp_times = np.arange(t_ss, timepoints[-1], shift)
    
    nbi = pop_array[t_ss_ind-1:, 1: max_m+1]
    #nvj = pop_array[t_ss_ind-1:, max_m+1 : 2*max_m+1]
    
    interp_fun_nbi = interp1d(timepoints.flatten(), nbi.toarray(), kind='linear', axis = 0)
    #interp_fun_nvj = interp1d(timepoints.flatten(), nvj.toarray(), kind='linear', axis = 0)
    nbi_interp = interp_fun_nbi(interp_times)
    #nvj_interp = interp_fun_nvj(interp_times)
    
    fraction_list = []

    for i in range(len(nbi_interp)):
        num_remaining = np.count_nonzero(nbi_interp[i:,np.where(nbi_interp[i] > 0)[0]], axis = 1)
        fraction_list.append(np.append(num_remaining / num_remaining[0], [np.nan]*i))
        
    turnover_array = np.stack(fraction_list)
    
    return(turnover_array, interp_times - t_ss)

def calculate_speed(turnover_array, interp_times):
    
    # calculate speed

    if np.nanmean(turnover_array, axis = 0)[-1] == 0: # if all clones eventually go extinct
        end_ind = np.where(np.nanmean(turnover_array, axis = 0) == 0)[0][0]

    else: # not everything goes extinct by the end, just use the last time point 
        end_ind = len(interp_times) - 1

    # use first 15% of curve and exclude first 1% to remove small spacer effects:
    t_ind = int(end_ind * 0.15)
    start_ind = int(end_ind * 0.01)
    speed = ((np.nanmean(turnover_array, axis = 0)[start_ind] - np.nanmean(turnover_array, axis = 0)[t_ind]) 
             / (interp_times[t_ind] - interp_times[start_ind]))
    
    return speed, start_ind

def effective_e(nbi, nvi, all_phages, pv_type, e, theta):
    """
    Calculate effective e for the different types of bacteria-phage interaction
    
    Inputs:
    nbi : array of sizes of bacterial clones of shape (time, max_m)
    nvi : array of sizes of phage clones of shape (time, max_m)
    pv_type : string describing pv(i,j) interaction type ('binary', 'exponential', 'exponential_025', or 'theta_function')
    all_phages : list of all protospacers corresponding to the columns of pop_array
    e : probability of matching spacer effectively blocking phage
    theta : pv theta function cutoff parameter if pv_type == 'theta_function'
    """
    
    # effective e parameter
    
    if pv_type == 'binary':
        e_effective_list = (e*np.sum(nbi*nvi, axis = 1)/
                        (np.array(np.sum(nvi, axis = 1)) * np.array(np.sum(nbi, axis = 1))))
    
    elif pv_type == 'exponential' or pv_type == 'exponential_025' or pv_type == 'theta_function':
        e_effective_list = []
        
        # make distance matrix
        all_phages = np.array(all_phages)
        distance_matrix = np.zeros((len(all_phages), len(all_phages)))
        for i in range(len(all_phages)):
            distance_matrix[i] = np.sum(np.abs(all_phages - all_phages[i]), axis = 1)
        
        if pv_type == 'exponential':
            for i in range(nbi.shape[0]):
                numerator = np.sum(np.outer(nbi[i], nvi[i]) * (1 -  e * np.exp(- distance_matrix)))
                denominator = np.sum(np.outer(nbi[i], nvi[i]))
                e_effective_list.append(1 - numerator / denominator)
        elif pv_type == 'exponential_025':
            for i in range(nbi.shape[0]):
                numerator = np.sum(np.outer(nbi[i], nvi[i]) * (1 -  e * np.exp(- 0.25* distance_matrix)))
                denominator = np.sum(np.outer(nbi[i], nvi[i]))
                e_effective_list.append(1 - numerator / denominator)
        elif pv_type == 'theta_function':
            distance_matrix_theta = np.copy(distance_matrix)
            distance_matrix_theta[distance_matrix <= theta] = (1-e)
            distance_matrix_theta[distance_matrix > theta] = 1
            for i in range(nbi.shape[0]):
                numerator = np.sum(np.outer(nbi[i], nvi[i])*distance_matrix_theta)
                denominator = np.sum(np.outer(nbi[i], nvi[i]))
                e_effective_list.append(1 - numerator / denominator)
                           
    return e_effective_list

def get_clone_sizes(pop_array, c0, e, max_m, t_ss_ind, pv_type, theta, all_phages, size_cutoff = 1, n_snapshots = 15):
    """
    Gets a size cutoff by choosing a size that gives the same mean number of phage clones
    as bacterial clones, then calculates the average time to extinction for phage trajectories starting
    at that cutoff.
    Also calculates mean m, mean e_effective, and mean nu.
    
    Inputs: 
    pop_array : pop_array in sparse format
    c0 : incoming nutrient concentration
    e : probability of matching spacer effectively blocking phage
    max_m : total number of unique protospacers in simulation
    t_ss_ind : index of when t > t_ss generations
    pv_type : string describing pv(i,j) interaction type ('binary', 'exponential', or 'exponential_025')
    all_phages : list of all protospacers corresponding to the columns of pop_array
    
    Returns:
    mean_m : mean number of bacteria clones
    size_cutoff : phage clone size at which mean number of phage clones equals mean number of bacteria clones
    np.nanmean(nbs/nb) : mean nu (mean fraction of bacteria with spacers)
    np.nanmean(e_effective_list) : mean effective e
    Delta_bac: effective number of bacteria alleles
    Delta_phage: effective number of phage alleles
    """
    
    # take n snapshots and calculate the mean clone size by combining all clones
    nbi = pop_array[t_ss_ind:, 1: max_m+1]
    nvj = pop_array[t_ss_ind:, max_m+1 : 2*max_m+1]
    nbs = np.array(np.sum(pop_array[t_ss_ind:, 1: max_m+1], axis = 1)).flatten()
    nb = np.array(np.sum(pop_array[t_ss_ind:, : max_m+1], axis = 1)).flatten()
    nv = np.array(np.sum(pop_array[t_ss_ind:, max_m + 1: 2*max_m+1], axis = 1)).flatten()
        
    nbi_snapshots = nbi[::int(len(nbs)/n_snapshots)].toarray()
    nvi_snapshots = nvj[::int(len(nv)/n_snapshots)].toarray()
    nbi_nan_snapshots = np.array(nbi_snapshots) 
    nbi_nan_snapshots[nbi_nan_snapshots == 0] = np.nan

    try:
        mean_clone_size = np.nanmean(nbi_nan_snapshots)
        mean_nbs = np.mean(nbs[::int(len(nbs)/n_snapshots)])
        mean_m = mean_nbs / mean_clone_size
    except:
        mean_m = np.nan
        
    try:
        mean_phage_m = np.nanmean(np.sum(nvi_snapshots > 0, axis = 1))
    except:
        mean_phage_m = np.nan
        
    try:
        mean_large_phage_m = np.nanmean(np.sum(nvi_snapshots > size_cutoff, axis = 1))
    except:
        mean_large_phage_m = np.nan
        
    try:
        mean_large_phage_size = np.nanmean(nvi_snapshots[nvi_snapshots > size_cutoff])
    except:
        mean_large_phage_size = np.nan
        
    # effective # of alleles (Kimura & Crow 1964)

    try:
        nbi_freqs = nbi_snapshots / nbs[::int(len(nbs)/n_snapshots), np.newaxis]
        Delta_bac = np.mean(1/np.sum(np.multiply(nbi_freqs, nbi_freqs), axis = 1))  # average across snapshots
    except:
        Delta_bac = np.nan
    try:
        nvi_freqs = nvi_snapshots / nv[::int(len(nv)/n_snapshots), np.newaxis]
        Delta_phage = np.mean(1/np.sum(np.multiply(nvi_freqs, nvi_freqs), axis = 1)) # average across snapshots
    except:
        Delta_phage = np.nan

    
    # effective e parameter
    
    e_effective_list = effective_e(nbi_snapshots, nvi_snapshots, all_phages, pv_type, e, theta)
    
    return mean_m, mean_phage_m, mean_large_phage_m, mean_large_phage_size, Delta_bac, Delta_phage, np.nanmean(nbs/nb), np.nanmean(e_effective_list)

def p_ext_virus(B, t, delta, N0):
    """
    An approximate solution for the probability of extinction for phage clones, valid at large t
    t is time in minutes
    This is the NEUTRAL APPROXIMATION (p = 1/B)
    """
    return ((B*delta*t)/(B*delta*t + 2))**N0

def p_ext_virus_phage_gens(B, t, N0):
    """
    An approximate solution for the probability of extinction for phage clones, valid at large t
    t is time in phage death generations. To convert to time in minutes, multiply
    t by delta.
    This is the NEUTRAL APPROXIMATION (p = 1/B)
    """
    return ((B*t)/(B*t + 2))**N0

def p_ext_virus_long_t(beta, delta, B, t, N0):
    """
    An approximate solution for the probability of extinction for phage clones, valid at large t
    t is time in minutes
    """
    p = beta/(beta+delta)
    s = beta*(B-1) - delta
    
    return ((2 - 3*B*p + p*B**2)*(1 - np.exp(s*t)) / (2 - B*p*(3- np.exp(s*t)) + p*(B**2)*(1 - np.exp(s*t))))**N0
    
    # version with time in phage generations instead of minutes
    #return (((-1 + np.exp((-1 + B*p)*t))*(2 + (-3 + B)*B*p))/(-2 + B*(3 - B + (-1 + B)*np.exp((-1 + B*p)*t))*p))**N0

def p_ext_virus_short_t(beta, delta, t,N0):
    """
    An approximate solution for the probability of extinction for phage clones, valid at short t
    t is time in phage death generations
    """
    return (delta*(1-np.exp(-t))/(beta+delta))**N0

def integrand(w, p, B):
    """
    To numerically calculate P_ext
    """
    return 1/(1-p+p*w**(B) - w)

def P0_time(zeta, p, B, limit = 50):
    """
    First output is integral
    Second output is upper bound on the error
    """
    from scipy.integrate import quad
    return quad(integrand, 0, zeta, args = (p, B), limit = limit)

vec_time = np.vectorize(P0_time) # !!!! THIS IS THE COOLEST FUNCTION AAAAHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH

def get_numeric_mean(P0_vals, time_vals, n_samples = 5000):
    """
    Estimate the mean of a cumulative distribution by sampling randomly from the distribution.
    
    time_vals should be in units of bacterial generations.
    """
    inds = []
    for n in np.random.rand(n_samples):
        inds.append(np.argmin(np.abs((1 - P0_vals) - n)))

    tvals = time_vals[inds]
    
    return np.mean(tvals) # in generations

def bacteria_clone_backwards_extinction(beta, delta, D, n, nbs):
    """
    Mean extinction time for bacterial clones,
    calculated using the backwards master equation corresponding to the 
    bacteria single clone master equation.
    This solution assumes that the drift term is zero,
    i.e. beta*n + D - delta*n = 0 (growth and acquisition balanced by death).
    This is not necessarily true, and depends on the clone size n being
    considered. I expect that it is approximately true for "large"
    bacterial clones that have reached their deterministic mean size.
    
    Inputs:
    beta : bacteria clone growth rate, beta = g*C
    delta : bacteria clone death rate, delta = F + r + alpha*pv*(nv - e*nvi)
    D : bacteria clone spacer acquisition rate, D = alpha*eta*nb0*nvi*(1-pv)
    n : bacteria clone size
    nbs : total number of bacteria with spacer
    
    Returns:
    mean_time_to_extinction : mean time to extinction in minutes from backwards master equation
    """
    
    mean_time_to_extinction = (2/((beta+delta)**2))*(D*np.log(D) - (D + (beta+delta)*n)*np.log(D + (beta+delta)*n) 
                           + (beta+delta)*n*(1+np.log(D + (beta+delta)*nbs)))
    
    return mean_time_to_extinction

def bacteria_backwards_master_changing_nvi(n, y, nbi_vec, nvi_vec, nb0, nv, C, g, c0, f, R, alpha, eta, pv, e):
    """
    Full backwards master equation for bacteria in fokker-planck form,
    including the drift term.
    In this version, we assume that nvi is a linearly increasing function of n,
    i.e. when n is small, nvi is also small. The ratio of nvi to n is the 
    ratio of nvi_ss to nbi_ss.
    
    Inputs:
    n : clone size to solve at
    y : y[0] = T[n], y[1] = dT / dn
    nbi_vec : vector of average population size of nbi over time (get this from numerical deterministic solution)
    nvi_vec : vector of average population size of nvi over time (get this from numerical deterministic solution)
    nb0 : average number of bacteria without spacers at steady-state
    nv : average phage population size at steady-state
    C : average nutrient concentration at steady-state
    g, c0, ... : simulation parameters
    """

    F = f*g*c0
    r = R*g*c0
    
    # write nvi as a function of nbi
    #nvi = np.interp(n, nbi_vec_new, nvi_vec_new)
    nvi = np.interp(n, nbi_vec, nvi_vec)

    b = g*C
    D_nvi = (alpha*eta*nb0*nvi)*(1-pv)
    d_nvi = F + r + alpha*pv*(nv - e*nvi)
    
    return np.vstack([y[1], 
            -2 * (1 + y[1]*(b*n + D_nvi - d_nvi*n))/(b*n + D_nvi + d_nvi*n)])

def bc(ya, yb):
    """
    Boundary conditions for bacteria_backwards_master
    
    Here a = 0, b = nbs
    
    Tn(0) = 0,
    dTn/dn (nbs) = 0
    
    """
    
    return np.array([ya[0], yb[1]])

def bac_extinction_time_numeric(nb_pred, nv_pred, C_pred, nb0_pred, nbs_pred, nvi_ss_pred, nbi_ss_pred, 
                                mean_bac_extinction_time, f, g, c0, e, alpha, B, mu, pv, R, eta):
    
    """
    Calculate numerical solution for bacteria extinction time.
    This uses the two-dimensional numerical solution for nvi and nbi to calculate the 
    dependence of nvi on nbi to integrate an extinction time for large bacterial clones.
    """
    
    # numerical time-dependent solution
    end_time = 5000 /(g*c0)
    t_eval = np.logspace(-1, np.log10(end_time), 500)[:-1]
    #t_eval = np.arange(0, end_time, 400)

    sol_nvi_nbi = solve_ivp(fun=lambda t, y: nvi_nbi_ode(t, y, nb_pred, nv_pred, C_pred, nb0_pred, f, 
                                                 g, c0, e, alpha, B, mu, pv, R, eta), 
                                t_span = [0, end_time], y0 = [1, 0], t_eval = t_eval, rtol = 10**-5)
    timevec1 = sol_nvi_nbi.t * g* c0
    nvi_vec1 = sol_nvi_nbi.y[0]
    nbi_vec1 = sol_nvi_nbi.y[1]
    
    # numerical P0(t) 

    F = f*g*c0
    n0 = 1
    beta = alpha*pv*nb_pred
    delta = F + alpha*nb_pred*(1-pv)
    p = beta/(beta+delta)
    s = beta*(B-1) - delta
    
    # get upper limit of zeta (for s > 0) from long-time approximation with s > 0
    max_zeta = 1 - (2*s / (B*(s+delta)))
    zeta_vals = (1-np.logspace(np.log10(1- max_zeta),0,100))[1:] # make zeta_vals end at the upper limit

    #times =  np.arange(0, 6000, 0.1) # in bacteria gens
    try:
        P_0_times =  vec_time(zeta_vals, p, B)[0]*g*c0/(alpha*nb_pred+F) # use p = beta / (beta + delta)
    except ZeroDivisionError: # for some reason some values of zeta trigger this error?
        return np.nan 
        
    P_0 = zeta_vals**n0

    P_0_interp = np.interp(timevec1, P_0_times[::-1], P_0[::-1]) # interp extends the edges by the last value
    
    # get approximate nvi_vec scaled by P0 with correct end behaviour
    try:
        ind2 = np.where(nvi_vec1 / (1-P_0_interp) > nvi_ss_pred)[0][0]
    except:
        ind2 = -1

    nvi_vec_new = np.append(nvi_vec1[:ind2] / (1-P_0_interp[:ind2]),[nvi_ss_pred])
    times_new = np.append(timevec1[:ind2], timevec1[-1]*10)
    nvi_vec_new = np.interp(timevec1, times_new, nvi_vec_new)

    # get piecewise nbi_vec scaled by P0
    try:
        ind2 = np.where(nbi_vec1 / (1-P_0_interp) > nbi_ss_pred)[0][0]
    except:
        ind2 = -1

    nbi_vec_new = nbi_vec1[:ind2] / (1-P_0_interp[:ind2])
    times_new = np.append(timevec1[:ind2], timevec1[-1]*10)
    nbi_vec_new = np.append(nbi_vec_new, nbi_ss_pred)
    nbi_vec_new = np.interp(timevec1, times_new, nbi_vec_new)
    
    # calculate numerical solution
    num_points = int(nbs_pred)
    x = np.linspace(0, nbs_pred, num_points) # range of the solution is n = 0 to n = nbs, these are the boundaries for bc
    y = np.zeros((2, x.size))

    # we expect Tn to be an increasing function of n
    nbi_ss_ind = find_nearest(x, nbi_ss_pred)
    max_y0 = (mean_bac_extinction_time/(g*c0) / nbi_ss_ind)*num_points

    # set the initial guess for Tn to be a linear function of n that passes through the approximate solution at nbi_ss
    # set dT/dn to be the gradient of the Tn guess
    y[0] = np.linspace(0, max_y0, num_points)
    y[1] = np.gradient(y[0], x)
    
    sol = solve_bvp(fun=lambda x, y: 
                 bacteria_backwards_master_changing_nvi(x, y, nbi_vec_new, nvi_vec_new, nb0_pred, nv_pred, 
                                                        C_pred, g, c0, f, R, alpha, eta, pv, e), 
                 bc=bc, x=x, y=y)
    
    mean_bac_extinction_time_changing_nvi = sol.y[0][find_nearest(sol.x, nbi_ss_pred)]*g*c0

    return mean_bac_extinction_time_changing_nvi

def bac_large_clone_extinction(pop_array, nbi, nvi, max_m, nbi_ss_pred, t_ss_ind):
    """
    Calculate bacterial extinction times for large bacterial clones
    Returns list of extinction times in minutes
    """
    t_all = pop_array[t_ss_ind:, -1].toarray().flatten()
    nbi_nonzero = pop_array[t_ss_ind:, 1: max_m+1] != 0
    
    # this new version keeps it sparse longer
    # captures transitions between False and True i.e. where nbi goes from 0 to >0
    rows_bac, cols_bac = np.where((nbi_nonzero[1:] > nbi_nonzero[:-1]).toarray()) 
    
    nvi_trajectories_bac_extinction = []
    nbi_trajectories_bac_extinction = []
    t_trajectories_bac_extinction = []
    bac_extinction_times_large = []
    bac_extinction_times_large_phage_present = []

    #nv = np.array(np.sum(pop_array[t_ss_ind:, max_m+1 : 2*max_m+1], axis = 1)).flatten()

    for i in range(len(cols_bac)):
        #nvi_traj = nvi[rows_bac[i]:, cols_bac[i]]    
        nbi_traj = nbi[rows_bac[i] :, cols_bac[i]]
        #t_i = t_all[rows_bac[i] : ]

        try: # split if multiple extinctions occur
            death_ind = np.where(nbi_traj == 0)[0][1] +1
            nvi_traj = nvi[rows_bac[i] : rows_bac[i] + death_ind, cols_bac[i]]
            t_i = t_all[rows_bac[i] : rows_bac[i] + death_ind]
            nbi_traj = nbi_traj[:death_ind]
        except:
            nvi_traj = nvi[rows_bac[i] :, cols_bac[i]]
            t_i = t_all[rows_bac[i] :]

        if np.any(nbi_traj >= nbi_ss_pred): 
            nvi_trajectories_bac_extinction.append(nvi_traj)
            nbi_trajectories_bac_extinction.append(nbi_traj)
            t_trajectories_bac_extinction.append(t_i)
            bac_establishment_ind = np.where(nbi_traj >= nbi_ss_pred)[0][0]
            try:
                bac_extinction_ind = np.where(nbi_traj[bac_establishment_ind:] == 0)[0][0]
                bac_extinction_times_large.append(t_i[bac_establishment_ind + bac_extinction_ind] 
                                                  - t_i[bac_establishment_ind])

                if nvi_traj[bac_establishment_ind + bac_extinction_ind] > 0: # then the phage is still present
                    bac_extinction_times_large_phage_present.append(t_i[bac_establishment_ind + bac_extinction_ind] 
                                                  - t_i[bac_establishment_ind])
            except: # if the trajectory doesn't go extinct
                pass
            
    return bac_extinction_times_large, bac_extinction_times_large_phage_present

def extinction_time(m, f, g, c0, alpha, B, pv, e, mu, R, eta):
    """
    Mean time to extinction for large phage clones (in bacterial generations)
    This is the neutral time to extinction from the phage clone 
    backwards master equation. The starting frequency is assumed to be the 
    deterministic mean phage clone size (nvi_ss).
    
    Inputs:
    m : mean number of bacterial clones at steady state
    other inputs : simulation parameters
        
    Outputs: a scalar value that is the mean time to extinction for large 
                phage clones behaving neutrally.
    """
        
    e_effective = e/m
    
    # get predicted mean field quantities
    nb, nv, C, nu = analytic_steady_state(pv, e_effective, B, R, eta, f, c0, g, alpha)
    nb0 = (1-nu)*nb
    
    nvi_ss = nvi_steady_state(nb, nv, C, nb0, f, g, c0, e, alpha, B, mu, pv, R, eta)
    nbi_ss = nbi_steady_state(nb, f, g, c0, e, alpha, B, mu, pv)
    
    F = f*g*c0
    freq = nvi_ss / nv
    beta = nb*alpha*pv - alpha*pv*e*nbi_ss
    delta = F + alpha*nb*(1-pv) + alpha*pv*e*nbi_ss
    
    return 2*nv*freq*(1-np.log(freq))*g*c0/((B-1)**2 * beta + delta)
    
def establishment_fraction(m, f, g, c0, alpha, B, pv, e, mu, R, eta):
    """
    Predicted fraction of new phage mutants that "establish" by reaching the
    deterministic mean clone size. This is the long-time limit of the survival
    probability for phage clones with a constant selection equal to the 
    initial fitness of a phage mutant. 
    """
    
    e_effective = e/m
    F = f*g*c0
    # get predicted mean field quantities
    nb, nv, C, nu = analytic_steady_state(pv, e_effective, B, R, eta, f, c0, g, alpha)
    
    beta_small = nb*alpha*pv
    delta_small = F + alpha*nb*(1-pv)
    
    p = beta_small / (beta_small + delta_small)
    return (1 - (2-3*B*p + p*B**2)/(B*p*(B-1)))
    
def mutation_rate(m, f, g, c0, alpha, B, pv, e, mu, R, eta):
    """
    Rate of creation of new phage clones (per minute).
    This is the mean-field phage reproduction rate multiplied by the 
    probability of one or more mutations per burst (1 - exp(-mu L)).
    This assumes that all clones are approximately equal in size ie. 
    nvi = nv/m
    """
    
    e_effective = e/m
    
    # get predicted mean field quantities
    nb, nv, C, nu = analytic_steady_state(pv, e_effective, B, R, eta, f, c0, g, alpha)
    
    L = 30
    P0 = np.exp(-mu*L)
    
    return alpha*B*(1-P0)*pv*nv*nb*(1-nu*e_effective)
    

def predict_m(m, f, g, c0, alpha, B, pv, e, mu, R, eta):
    """
    Calculate a predicted m given an input m. The idea is that we can iterate until they match.
    
    Inputs:
    parameters, m
    
    Outputs: 
    predicted m
    predicted establishment fraction
    predicted mutation rate
    predicted mean time to extinction for large clones
    """
    
    mean_T_backwards = extinction_time(m, f, g, c0, alpha, B, pv, e, mu, R, eta)

    # numeric mean of cumulative dist
    #times =  np.arange(0, 100000, 0.1) # in bacterial generations
    #mean_P0_large_s0 = get_numeric_mean(p_ext_virus(B, times/(g*c0), delta, nvi_ss), times, n_samples = 1000) 
    
    # numerically integrate to a large t, assuming s < 0
    #s = beta*(B*P0-1) - delta
    #t_max = 10**8
    #mean_time_numeric, error = mean_time(t_max, -np.abs(s), B, delta, nvi_ss)
    #mean_P0_large_s0 = mean_time_numeric*g*c0
    
    predicted_establishment_fraction = establishment_fraction(m, f, g, c0, alpha, B, pv, e, mu, R, eta)
    mutation_rate_pred = mutation_rate(m, f, g, c0, alpha, B, pv, e, mu, R, eta)
    
    pred_m = predicted_establishment_fraction * (mutation_rate_pred / (g*c0)) * mean_T_backwards
    #pred_m = predicted_establishment_fraction * (mutation_rate_pred / (g*c0)) * mean_P0_large_s0
    
    return pred_m, predicted_establishment_fraction, mutation_rate_pred / (g*c0), mean_T_backwards
   
def m_dot(m, f, g, c0, alpha, B, pv, e, mu, R, eta):
    """
    Calculate a predicted rate of change of m given an input m. 
    
    Inputs:
    parameters, m
    
    Outputs: 
    rate of change of m (in units of bacterial generations)
    """
    
    mean_T_backwards = extinction_time(m, f, g, c0, alpha, B, pv, e, mu, R, eta)
    
    predicted_establishment_fraction = establishment_fraction(m, f, g, c0, alpha, B, pv, e, mu, R, eta)
    mutation_rate_pred = mutation_rate(m, f, g, c0, alpha, B, pv, e, mu, R, eta)
    
    return predicted_establishment_fraction * (mutation_rate_pred / (g*c0)) - m/mean_T_backwards    

def recursive_bac_m(m_vals_test, f, g, c0, alpha, B, pv, e, mu, R, eta):
    
    """
    Predict bacteria m recursively using a range of input m and finding the 
    intersection between the input and output m.
    
    m_vals_test : the range of m to iterate over
    """
    
    #m_vals_test = np.arange(target_m*0.1, target_m*5, float(target_m * 0.1))
    
    pred_m_list = []
    establishment_fraction_list = []
    mutation_rate_list = []
    extinction_time_list = []
    m_test = []
    
    for m in m_vals_test:
        
        if e/m > 1: # this is a non-physical case to test
            continue
            
        m_test.append(m)        
        pred_m, predicted_establishment_fraction, mutation_rate_pred, mean_P0_large_s0 = predict_m(m, 
                                                                            f, g, c0, alpha, B, pv, e, mu, R, eta)
        pred_m_list.append(pred_m)
        establishment_fraction_list.append(predicted_establishment_fraction)
        mutation_rate_list.append(mutation_rate_pred)
        extinction_time_list.append(mean_P0_large_s0)

    m_test = np.array(m_test)
    
    # get right-most intersection - the stable fixed point
    try:
        m_ind = np.where((np.array(pred_m_list).flatten() - m_test)[::-1] > 0)[0][0]
        pred_m_final = m_test[::-1][m_ind]
        establishment_fraction_final = establishment_fraction_list[::-1][m_ind]
        mutation_rate_final = mutation_rate_list[::-1][m_ind]
        extinction_time_final = extinction_time_list[::-1][m_ind]
    except IndexError: # no intersection
        pred_m_final = np.nan 
        establishment_fraction_final = np.nan
        mutation_rate_final = np.nan
        extinction_time_final = np.nan
        
    return pred_m_final, establishment_fraction_final, mutation_rate_final, extinction_time_final

if __name__ == "__main__":
    pass
    
