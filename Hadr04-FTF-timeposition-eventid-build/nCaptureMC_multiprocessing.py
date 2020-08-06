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
#from scipy import optimize
from scipy.interpolate import griddata
import random
from scipy import stats
from multiprocessing import Pool

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
    
    # Split x and y data
    points = dfenergyncap[['energy','ncapcount']].to_numpy()
    values = dfenergyncap['eventcount'].to_numpy()

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

    m1, m2, m3 = dfe.loc[dfe['energy'] == en]['rho'], dfe.loc[dfe['energy'] == en]['z'], dfe.loc[dfe['energy'] == en]['t']
    xmin = dfe.loc[dfe['energy'] == en]['rho'].min()
    xmax = dfe.loc[dfe['energy'] == en]['rho'].max()
    ymin = dfe.loc[dfe['energy'] == en]['z'].min()
    ymax = dfe.loc[dfe['energy'] == en]['z'].max()
    tmin = dfe.loc[dfe['energy'] == en]['t'].min()
    tmax = dfe.loc[dfe['energy'] == en]['t'].max()

    X, Y, T = np.mgrid[xmin:xmax:100j, ymin:ymax:100j, tmin:tmax:100j] # makes 100 by 100 grid
    positions = np.vstack([X.ravel(), Y.ravel(), T.ravel()])
    values = np.vstack([m1, m2, m3])
    kernel = stats.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    # Generate the bins for each axis
    x_bins = np.linspace(xmin, xmax, Z.shape[0]+1)
    y_bins = np.linspace(ymin, ymax, Z.shape[1]+1)
    t_bins = np.linspace(tmin, tmax, Z.shape[1]+1)

    # Find the middle point for each bin
    x_bin_midpoints = x_bins[:-1] + np.diff(x_bins)/2
    y_bin_midpoints = y_bins[:-1] + np.diff(y_bins)/2
    t_bin_midpoints = t_bins[:-1] + np.diff(t_bins)/2

    # Calculate the Cumulative Distribution Function(CDF)from the PDF
    cdf = np.cumsum(Z.ravel())
    cdf = cdf / cdf[-1] # Normalisation

    # Create random data
    values = np.random.rand(num)

    # Find the data position
    value_bins = np.searchsorted(cdf, values)
    x_idx, y_idx, t_idx = np.unravel_index(value_bins,
                                    (len(x_bin_midpoints),
                                    len(y_bin_midpoints),
                                    len(t_bin_midpoints)))

    # Create the new data
    new_data = np.column_stack((x_bin_midpoints[x_idx],
                                y_bin_midpoints[y_idx],
                                t_bin_midpoints[t_idx]))
    new_x, new_y, new_z = new_data.T

    return new_data

# %%
def interp_rho_z_t(energy, num=1):

    """
    Gives samples of rho and z at given energy based on values
    interpolated from data.
    """

    loc_left = np.searchsorted(energy_list, energy, side='right')-1
    loc_right = np.searchsorted(energy_list, energy)
    en1 = energy_list[loc_left]
    en2 = energy_list[loc_right]


    if __name__ == '__main__':
        p = Pool(2)
        results = p.starmap(sample_at_data_energy, [(en1, num), (en2, num)])

    samplearr1 = results[0]
    samplearr2 = results[1]
    i = 0
    list_out_rho_z = []
    while i < np.shape(samplearr1)[0]:
        new_data1 = samplearr1[i]
        new_data2 = samplearr2[i]
        values_to_interp = np.vstack([new_data1, new_data2])#.transpose()
        list_out_rho_z.append(griddata([en1, en2], values_to_interp, energy, method='linear', rescale=True))
        i+=1

    df_out_rho_z = pd.DataFrame(np.row_stack(list_out_rho_z), columns=['rho', 'z', 't'])

    return df_out_rho_z



# %%
def ncap_sim(energy, num_n=1):
    
    # Get number of ncaps
    num_ncaps = sum(ncap_num(energy, num_n))
    # Get positions
    if energy in energy_list:
        return pd.DataFrame(sample_at_data_energy(energy, num_ncaps), columns=['rho', 'z', 't'])
    else:
        return interp_rho_z_t(energy, num_ncaps)

# %%
import time 

t0 = time.time()
dfresults=ncap_sim(1875.3, 2000)
print(ncap_sim(1875.3, 2000))
t1 = time.time()

total = t1-t0
print('Execution time: ', total)

# %%
plt.scatter(dfresults['rho'],dfresults['z'])

