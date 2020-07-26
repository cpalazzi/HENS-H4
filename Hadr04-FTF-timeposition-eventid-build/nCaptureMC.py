#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 09:13:25 2020

@author: carlopalazzi
"""

# %%
# %matplotlib auto
# Sets plots to appear in separate window
# Comment out and restart kernel to reset to inline plots

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('text', usetex=False)
from scipy import optimize
from scipy.interpolate import griddata
import random
from scipy import stats

# %%
# Read count and positions datasets
dfenergyncap = pd.read_csv('dfenergyncap.csv')
dfe = pd.read_csv('dfe_cylindrical.csv')

# List (array) of energies in data
energy_list = dfe['energy'].unique()

# %%
# Pairwise correlation matrix heatmap
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 6))
corr = dfe.corr(method='kendall')
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                 linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap', fontsize=14)
plt.savefig('images/correlation.png', dpi=800, bbox_inches='tight')
plt.show()

# %%
def ncap_num(energy, num_n=1):
        num_n_cap = range(40)
        energy,num_n_cap = np.meshgrid(energy,num_n_cap)
        # interpolate
        grid_z0 = griddata(points, values, (energy,num_n_cap), method='linear')
        # Replace nans with 0
        grid_z0 = np.nan_to_num(grid_z0)
        # Calculate area under interpolation at given energy
        area = np.trapz(grid_z0,num_n_cap,axis=0)
        # Normalise
        grid_z0_norm = grid_z0/area
        # Sample
        choicelist = random.choices(num_n_cap, weights=grid_z0_norm, k=num_n)
        choicelist = [int(i) for i in choicelist]

        return choicelist


# %%
# Function to sample from dataset at given energy
def sample_at_data_energy(en, num=1):
    
    """
    Function that generates sample of rho at z at given energy
    from stored data.
    """

    m1, m2 = dfe.loc[dfe['energy'] == en]['rho'], dfe.loc[dfe['energy'] == en]['z']
    xmin = dfe.loc[dfe['energy'] == en]['z'].min()
    xmax = dfe.loc[dfe['energy'] == en]['z'].max()
    ymin = dfe.loc[dfe['energy'] == en]['rho'].min()
    ymax = dfe.loc[dfe['energy'] == en]['rho'].max()

    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j] # makes 100 by 100 grid
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Generate the bins for each axis
    x_bins = np.linspace(xmin, xmax, Z.shape[0]+1)
    y_bins = np.linspace(ymin, ymax, Z.shape[1]+1)

    # Find the middle point for each bin
    x_bin_midpoints = x_bins[:-1] + np.diff(x_bins)/2
    y_bin_midpoints = y_bins[:-1] + np.diff(y_bins)/2

    # Calculate the Cumulative Distribution Function(CDF)from the PDF
    cdf = np.cumsum(Z.ravel())
    cdf = cdf / cdf[-1] # Normalisation

    # Create random data
    values = np.random.rand(num)

    # Find the data position
    value_bins = np.searchsorted(cdf, values)
    x_idx, y_idx = np.unravel_index(value_bins,
                                    (len(x_bin_midpoints),
                                    len(y_bin_midpoints)))

    # Create the new data
    new_data = np.column_stack((x_bin_midpoints[x_idx],
                                y_bin_midpoints[y_idx]))
    new_x, new_y = new_data.T

    return new_data

# %%
def sample_rho_z(energy, num=1):

    """
    Gives samples of rho and z at given energy based on values
    interpolated from data.
    """

    loc_left = np.searchsorted(energy_list, energy, side='right')-1
    loc_right = np.searchsorted(energy_list, energy)
    en1 = energy_list[loc_left]
    en2 = energy_list[loc_right]

    samplearr1 = sample_at_data_energy(en1,num)
    samplearr2 = sample_at_data_energy(en2,num)
    i = 0
    listOutRhoZ = []
    while i < np.shape(samplearr1)[0]:
        new_data1 = samplearr1[i]
        new_data2 = samplearr2[i] 
        valuestointerp = np.vstack([new_data1, new_data2]).transpose()
        listOutRhoZ.append(griddata([en1, en2], valuestointerp, energy, method='linear', rescale=True))
        i+=1

    dfOutRhoZ = pd.DataFrame(listOutRhoZ, columns=['rho', 'z'])

    return dfOutRhoZ


# %%
def ncap_sim(energy, num_n=1):
    
    # Get number of ncaps
    num_ncaps = sum(ncap_num(energy, num_n))
    # Get positions
    return sample_rho_z(energy, num_ncaps)

# %%
ncap_sim(1666)

# %%
