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
import numpy.random as ra
from scipy.optimize import minimize
from scipy.special import factorial
from scipy import interpolate


# %%
# Read and combine datasets
dfenergyncap = pd.read_csv('dfenergyncap.csv')
# %%
points = dfenergyncap[['energy','ncapcount']].to_numpy()
values = dfenergyncap['eventcount'].to_numpy()
print(points)
print(values)
# %%
# LinearND interpolation
from scipy.interpolate import LinearNDInterpolator
myInterpolator = LinearNDInterpolator(points, values)
# %%
x = np.linspace(0,2000,50)
y = np.linspace(0,40,50)
x,y = np.meshgrid(x,y)
z = myInterpolator(x,y)
print(len(x))
print(len(y))
print(len(z))
# %%
print(z)

plt.plot(x, z)
plt.show()

# %%
# 3D scatter plot LinearND interpolation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Number of nCapture in event')
ax.set_zlabel('Number of events')
plt.show()

# %%
# griddata interpolation
x = np.linspace(0,2000,1000)
y = np.linspace(0,40,1000)
x,y = np.meshgrid(x,y)
from scipy.interpolate import griddata
grid_z0 = griddata(points, values, (x,y), method='linear')
# %%

plt.plot(x, grid_z0)
plt.show()

# %%
# %%
# 3D scatter plot griddata interpolation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, grid_z0)
ax.set_xlabel('Energy (MeV)')
ax.set_ylabel('Number of nCapture in event')
ax.set_zlabel('Number of events')
plt.show()

# %%
# griddata interpolation
energy = 11
NumNCap = np.linspace(0,40,10000)
energy,NumNCap = np.meshgrid(energy,NumNCap)

from scipy.interpolate import griddata
# interpolate
grid_z0 = griddata(points, values, (energy,NumNCap), method='linear')
# Replace nans with 0
grid_z0 = np.nan_to_num(grid_z0)
# %%

plt.plot(NumNCap, grid_z0)
plt.xlabel('Number of nCap in event')
plt.ylabel('Number of events')
plt.show()

# %%
area = np.trapz(grid_z0,NumNCap,axis=0)

# %%
# Normalise
grid_z0 = grid_z0/area

# %%
# Check normalisation
areacheck = np.trapz(grid_z0,NumNCap,axis=0)
print('areacheck =', areacheck)

plt.plot(NumNCap, grid_z0)
plt.xlabel('Number of nCap in event')
plt.ylabel('Number of events')
plt.title('Normalised Slice at Set Energy')
plt.show()

# Next need to figure out how to sample from pdf at each energy but making
# number of events whole number
# I'm thinking Choices are on x axis, weights on y

# %%
def ncapsim(numn,energy):
        int numncap = 0
        for i in range(numn):

                grid_z = griddata(points, values, (x,energy), method='linear')
        return numncap

# %%
# Specify an energy to sample at. Get random numncap using randn/choice
# This returns number of events, which is proportional to a probability. 
# Look into rejection sampling.
# Each energy slice is an unnormalised pdf. 