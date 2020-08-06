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
# Read and combine datasets
df1 = pd.read_csv('timepositions1.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df2 = pd.read_csv('timepositions2.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df3 = pd.read_csv('timepositions3.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df4 = pd.read_csv('timepositions4.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df5 = pd.read_csv('timepositions5.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df6 = pd.read_csv('timepositions6.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df7 = pd.read_csv('timepositions7.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df8 = pd.read_csv('timepositions8.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df9 = pd.read_csv('timepositions9.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df10 = pd.read_csv('timepositions10.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df11 = pd.read_csv('timepositions11.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df12 = pd.read_csv('timepositions12.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df13 = pd.read_csv('timepositions13.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df14 = pd.read_csv('timepositions14.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df15 = pd.read_csv('timepositions15.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df16 = pd.read_csv('timepositions16.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df17 = pd.read_csv('timepositions17.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df18 = pd.read_csv('timepositions18.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df19 = pd.read_csv('timepositions19.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df20 = pd.read_csv('timepositions20.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df30 = pd.read_csv('timepositions30.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df40 = pd.read_csv('timepositions40.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df50 = pd.read_csv('timepositions50.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df60 = pd.read_csv('timepositions60.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df70 = pd.read_csv('timepositions70.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df80 = pd.read_csv('timepositions80.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df90 = pd.read_csv('timepositions90.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df100 = pd.read_csv('timepositions100.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df200 = pd.read_csv('timepositions200.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df300 = pd.read_csv('timepositions300.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df400 = pd.read_csv('timepositions400.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df500 = pd.read_csv('timepositions500.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df600 = pd.read_csv('timepositions600.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df700 = pd.read_csv('timepositions700.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df800 = pd.read_csv('timepositions800.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df900 = pd.read_csv('timepositions900.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df1000 = pd.read_csv('timepositions1000.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])
df2000 = pd.read_csv('timepositions2000.csv', usecols=[0,1,2,3], names=['t', 'x', 'y', 'z'])

dflist = [df10,df20,df30,df40,df50,df60,df70,df80,df90,df100,\
    df200,df300,df400,df500,df600,df700,df800,df900,df1000,df2000]

# Convert units to metres and microseconds
for df in dflist: 
    df[['t','x','y','z']] = df[['t','x','y','z']]/1000



# %%
dfcouples = [(10,df10)\
        ,(20,df20)\
        ,(30,df30)\
        ,(40,df40)\
        ,(50,df50)\
        ,(60,df60)\
        ,(70,df70)\
        ,(80,df80)\
        ,(90,df90)\
        ,(100,df100)\
        ,(200,df200)\
        ,(300,df300)\
        ,(400,df400)\
        ,(500,df500)\
        ,(600,df600)\
        ,(700,df700)\
        ,(800,df800)\
        ,(900,df900)\
        ,(1000,df1000)\
        ,(2000,df2000)]

# %%
# Add energy column to all dataframes
for energy, dfc in dfcouples:
        dfc['energy']=energy

# Concatenate all dataframes
dfe = pd.concat(dflist, ignore_index=True)

dfe
# %%
# Reorder columns
dfe = dfe[['energy','t','x','y','z']]

dfe
# %%
# Create cylindrical coordinates columns
# def cyltheta(row):
#     if row['x'] == 0:
#         val = 0
#     else:
#         val = np.arctan(row['y']/row['x'])
#     return val

dfe['rho']=np.sqrt(dfe['x']**2+dfe['y']**2)
dfe['theta']=np.arctan(dfe['y']/dfe['x'])
dfe['theta'].fillna(0, inplace=True) # Replace NaN theta with 0

dfe

# %%
# Save dataframe with cylindrical coordinates as csv
dfe.to_csv('dfe_cylindrical.csv',index=False)

# %%
# # 3D scatterplot of cylindrical data
# x = dfe['rho']
# y = dfe['theta']
# z = dfe['z']
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib import cm
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x, y, z)
# ax.set_xlabel('rho')
# ax.set_ylabel('theta')
# ax.set_zlabel('z')
# #ax.set_markersize(0.1)
# plt.show()

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
# Split x and y data
points = dfe['energy'].to_numpy()
values = dfe[['energy','t','x','y','z']].to_numpy()

eeval = 346
teval = np.linspace(0,500,10)
poseval = np.linspace(0,5,10)
evals = np.meshgrid(eeval,teval,poseval,poseval,poseval)
# interpolate
grid_z0 = griddata(points, values, evals, method='linear', rescale=True)

print(grid_z0)

# %%
print(np.arctan(0.1679/(-1.523)))

# %%
pd.DataFrame(data=grid_z0[1:,1:]#,    # values
            #index=data[1:,0],    # 1st column as index
            #columns=grid_z0[0,1:]
            )
# %%
x = dfe['x']
plt.plot(x,grid_z0)
plt.show()

# %%
plt.scatter(dfe['rho'],dfe['z'],s=0.1)
plt.xlabel('rho (m)')
plt.ylabel('z (m)')
plt.savefig('images/zandrhoPositions.png', dpi=800, bbox_inches='tight')
plt.show()

# %%
print(np.shape(dfe[['energy','rho','z']].transpose()))
# %%

density10 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 10][['rho','z']].transpose())
density20 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 20][['rho','z']].transpose())
density30 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 30][['rho','z']].transpose())
density40 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 40][['rho','z']].transpose())
density50 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 50][['rho','z']].transpose())
density60 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 60][['rho','z']].transpose())
density70 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 70][['rho','z']].transpose())
density80 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 80][['rho','z']].transpose())
density90 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 90][['rho','z']].transpose())
density100 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 100][['rho','z']].transpose())
density200 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 200][['rho','z']].transpose())
density300 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 300][['rho','z']].transpose())
density400 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 400][['rho','z']].transpose())
density500 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 500][['rho','z']].transpose())
density600 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 600][['rho','z']].transpose())
density700 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 700][['rho','z']].transpose())
density800 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 800][['rho','z']].transpose())
density900 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 900][['rho','z']].transpose())
density1000 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 1000][['rho','z']].transpose())
density2000 = stats.kde.gaussian_kde(dfe.loc[dfe['energy'] == 2000][['rho','z']].transpose())

# %%
# Take grid sample from each density
evalarr = np.array([[np.linspace(dfe['rho'].min(), dfe['rho'].max(), 100)]
            ,[np.linspace(dfe['z'].min(), dfe['z'].max(), 100)]
            ,[np.linspace(dfe['t'].min(), dfe['t'].max(), 100)]]).transpose()

rhovalues = np.array(np.linspace(dfe['rho'].min(), dfe['rho'].max(), 100))
zvalues = np.array(np.linspace(dfe['z'].min(), dfe['z'].max(), 100))
#tvalues = np.array(np.linspace(dfe['t'].min(), dfe['t'].max(), 100))

(rhovalues, zvalues) = np.meshgrid(rhovalues, zvalues)

values = np.vstack([rhovalues.ravel(), zvalues.ravel()])

print(density10(values))

# %%
print(np.shape(dfe.loc[dfe['energy'] == 10][['rho','z']].transpose()))
# 1D case
energy = 127
NumNCap = range(40)
energy,NumNCap = np.meshgrid(energy,NumNCap)
# interpolate
grid_z0 = griddata(points, values, (energy,NumNCap), method='linear')
# %%
# Testing 2D kde sampling code
en = 300

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

# fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#           extent=[xmin, xmax, ymin, ymax])
# ax.plot(m1, m2, 'k.', markersize=2)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])


# %%
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
values = np.random.rand(1)

# Find the data position
value_bins = np.searchsorted(cdf, values)
x_idx, y_idx = np.unravel_index(value_bins,
                                (len(x_bin_midpoints),
                                 len(y_bin_midpoints)))

# Create the new data
new_data1 = np.column_stack((x_bin_midpoints[x_idx],
                            y_bin_midpoints[y_idx]))
new_x, new_y = new_data1.T

print(new_data1)

# # %%
# kernel = stats.gaussian_kde(new_data.T)
# new_Z = np.reshape(kernel(positions).T, X.shape)

# fig, ax = plt.subplots()
# ax.imshow(np.rot90(new_Z), cmap=plt.cm.gist_earth_r,
#           extent=[xmin, xmax, ymin, ymax])
# ax.plot(new_x, new_y, 'k.', markersize=2)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])

# %%
# Run again at energy above
# Testing 2D kde sampling code
en = 400

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

# fig, ax = plt.subplots()
# ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
#           extent=[xmin, xmax, ymin, ymax])
# ax.plot(m1, m2, 'k.', markersize=2)
# ax.set_xlim([xmin, xmax])
# ax.set_ylim([ymin, ymax])


# %%
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
values = np.random.rand(1)

# Find the data position
value_bins = np.searchsorted(cdf, values)
x_idx, y_idx = np.unravel_index(value_bins,
                                (len(x_bin_midpoints),
                                 len(y_bin_midpoints)))

# Create the new data
new_data2 = np.column_stack((x_bin_midpoints[x_idx],
                            y_bin_midpoints[y_idx]))
new_x, new_y = new_data.T

print(new_data)


# %%
valuestointerp = np.vstack([new_data1, new_data2]).transpose()
print(valuestointerp)

# %%
# Now need to interpolate the values
grid_z0 = griddata([300, 400], valuestointerp, 345, method='linear', rescale=True)

print(grid_z0)


# %%
# from sympy.solvers import solve
# from sympy import Symbol
# y2=1.15644585e-01
# y1=6.26922222e-01
# x2=100
# x1=200
# en=127
# y = Symbol('y')

# solve((y-y1)/(y2-y1)-(en-x1)/(x2-x1), y)



# %%
