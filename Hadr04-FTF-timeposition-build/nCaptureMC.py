#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:13:25 2020

@author: carlopalazzi
"""

# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('text', usetex=False)
from scipy import optimize
import numpy.random as ra
# %%
# Read and combine datasets
df1 = pd.read_csv('timepositions1.csv', names=['t', 'x', 'y', 'z'])
df2 = pd.read_csv('timepositions2.csv', names=['t', 'x', 'y', 'z'])
df3 = pd.read_csv('timepositions3.csv', names=['t', 'x', 'y', 'z'])
df4 = pd.read_csv('timepositions4.csv', names=['t', 'x', 'y', 'z'])
df5 = pd.read_csv('timepositions5.csv', names=['t', 'x', 'y', 'z'])
df6 = pd.read_csv('timepositions6.csv', names=['t', 'x', 'y', 'z'])
df7 = pd.read_csv('timepositions7.csv', names=['t', 'x', 'y', 'z'])
df8 = pd.read_csv('timepositions8.csv', names=['t', 'x', 'y', 'z'])
df9 = pd.read_csv('timepositions9.csv', names=['t', 'x', 'y', 'z'])
df10 = pd.read_csv('timepositions10.csv', names=['t', 'x', 'y', 'z'])
df11 = pd.read_csv('timepositions11.csv', names=['t', 'x', 'y', 'z'])
df12 = pd.read_csv('timepositions12.csv', names=['t', 'x', 'y', 'z'])
df13 = pd.read_csv('timepositions13.csv', names=['t', 'x', 'y', 'z'])
df14 = pd.read_csv('timepositions14.csv', names=['t', 'x', 'y', 'z'])
df15 = pd.read_csv('timepositions15.csv', names=['t', 'x', 'y', 'z'])
df16 = pd.read_csv('timepositions16.csv', names=['t', 'x', 'y', 'z'])
df17 = pd.read_csv('timepositions17.csv', names=['t', 'x', 'y', 'z'])
df18 = pd.read_csv('timepositions18.csv', names=['t', 'x', 'y', 'z'])
df19 = pd.read_csv('timepositions19.csv', names=['t', 'x', 'y', 'z'])
df20 = pd.read_csv('timepositions20.csv', names=['t', 'x', 'y', 'z'])
df30 = pd.read_csv('timepositions30.csv', names=['t', 'x', 'y', 'z'])
df40 = pd.read_csv('timepositions40.csv', names=['t', 'x', 'y', 'z'])
df50 = pd.read_csv('timepositions50.csv', names=['t', 'x', 'y', 'z'])
df60 = pd.read_csv('timepositions60.csv', names=['t', 'x', 'y', 'z'])
df70 = pd.read_csv('timepositions70.csv', names=['t', 'x', 'y', 'z'])
df80 = pd.read_csv('timepositions80.csv', names=['t', 'x', 'y', 'z'])
df90 = pd.read_csv('timepositions90.csv', names=['t', 'x', 'y', 'z'])
df100 = pd.read_csv('timepositions100.csv', names=['t', 'x', 'y', 'z'])
df200 = pd.read_csv('timepositions200.csv', names=['t', 'x', 'y', 'z'])
df300 = pd.read_csv('timepositions300.csv', names=['t', 'x', 'y', 'z'])
df400 = pd.read_csv('timepositions400.csv', names=['t', 'x', 'y', 'z'])
df500 = pd.read_csv('timepositions500.csv', names=['t', 'x', 'y', 'z'])
df600 = pd.read_csv('timepositions600.csv', names=['t', 'x', 'y', 'z'])
df700 = pd.read_csv('timepositions700.csv', names=['t', 'x', 'y', 'z'])
df800 = pd.read_csv('timepositions800.csv', names=['t', 'x', 'y', 'z'])
df900 = pd.read_csv('timepositions900.csv', names=['t', 'x', 'y', 'z'])
df1000 = pd.read_csv('timepositions1000.csv', names=['t', 'x', 'y', 'z'])
df2000 = pd.read_csv('timepositions2000.csv', names=['t', 'x', 'y', 'z'])

dflist = [df1,df2,df3,df4,df5,df6,df7,df8,df9,df10,df11,df12,df13,df14,df15,\
          df16,df17,df18,df19,df20,df30,df40,df50,df60,df70,df80,df90\
          ,df100,df200,df300,df400,df500,df600,df700,df800,df900,df1000\
          ,df2000]

# Convert units to metres and microseconds
for df in dflist: 
    df[df.columns] = df[df.columns]/1000

# Create list of energies
energies = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19\
                     ,20,30,40,50,60,70,80,90\
            ,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000])
    
# Add energy column to each df
for i in range(len(dflist)):
    dflist[i]['energy']=energies[i]

# Append all dfs into one
dfall = df10.append(dflist)
    
# Move energy column to front
dfall = dfall[ ['energy'] + [ col for col in df.columns if col != 'energy' ] ]
# %% 
# Create df with counts
dfcounts = dfall.groupby(['energy']).size().reset_index(name="count")
print(dfcounts)
beamnumber=2000
counts = dfcounts['count']/beamnumber

# %%
from scipy.interpolate import interp1d
f = interp1d(energies,counts, kind='quadratic')

enew = range(energies[0],energies[-1]+1)
plt.plot(energies, counts, 'o', enew, f(enew), '--')
plt.xlim(left=0, right=20)
plt.ylim(top=3)
plt.xlabel('Energy (MeV)')
plt.ylabel('Count')
plt.show()