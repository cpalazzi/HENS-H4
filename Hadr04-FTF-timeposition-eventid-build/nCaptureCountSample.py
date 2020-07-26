#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:13:25 2020

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

# %%
# Read and combine datasets
dfenergyncap = pd.read_csv('dfenergyncap.csv')
dfenergyncap
# %%
# Split x and y data
points = dfenergyncap[['energy','ncapcount']].to_numpy()
values = dfenergyncap['eventcount'].to_numpy()
# %%
def ncapsim(energy, numn=1):
        NumNCap = range(40)
        energy,NumNCap = np.meshgrid(energy,NumNCap)
        # interpolate
        grid_z0 = griddata(points, values, (energy,NumNCap), method='linear')
        # Replace nans with 0
        grid_z0 = np.nan_to_num(grid_z0)
        # Calculate area under interpolation at given energy
        area = np.trapz(grid_z0,NumNCap,axis=0)
        # Normalise
        grid_z0_norm = grid_z0/area
        # Sample
        choicelist = random.choices(NumNCap, weights=grid_z0_norm, k=numn)
        choicelist = [int(i) for i in choicelist]

        return choicelist

# %%
ncapsim(437, 2)
# %%
# Plot sampler result at one energy

# plt.scatter(NumNCap, ncapsim())
# plt.xlabel('Number of nCap in event')
# plt.ylabel('Number of events')
# plt.title('Normalised Slice at Set Energy')
# plt.show()

# %%
# Plot sampler results at lots of energies
energies = list(range(10,2000,10))

testdf = pd.DataFrame(columns=['energy','ncapcount'])

numn = 100
for i in energies:
        elist = numn*[i]
        ncapcountlist = ncapsim(i,numn)
        newrows = pd.DataFrame(np.column_stack([elist, ncapcountlist]),columns=['energy','ncapcount'])
        testdf = testdf.append(newrows, ignore_index=True)

# Count number of events with each ncapcount
testdf = testdf.groupby(['energy','ncapcount']).size().reset_index()
testdf.columns = ['energy','ncapcount','eventcount']
# Make eventcount independent of number of runs
testdf['eventcount']=testdf['eventcount']/numn

# %%
# 3D scatterplot of generated data
x = testdf['energy']
y = testdf['ncapcount']
z = testdf['eventcount']
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, s=1)
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Number of nCapture in event')
ax.set_ylim(-1,40)
ax.set_zlabel('Number of events')
plt.savefig('images/Sampled_nCapCount_3D.png', dpi=800)
plt.show()

# %%
