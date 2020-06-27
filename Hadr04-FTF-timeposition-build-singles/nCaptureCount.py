#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:52:38 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

df10 = pd.read_csv('timepositions10.csv', names=['t', 'x', 'y', 'z'])
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

dflist = [df10, df20, df30, df40, df50, df60, df70, df80, df90\
          , df100, df200, df300, df400, df500, df600, df700, df800, df900, df1000\
          , df2000]

# Convert units to metres and microseconds
for df in dflist: 
    df[df.columns] = df[df.columns]/1000

entries = []
for df in dflist:
    entries.append(len(df))

# Create list of energies
energies = [10,20,30,40,50,60,70,80,90\
            ,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]

# Create dataframe of energies and counts
dfcount = pd.DataFrame(list(zip(energies, entries)), 
               columns =['energy', 'count']) 

# Create test function
def test_func(e, m, b):
    return np.multiply(m,e)+b

# Fitting the count as a function of the energy ############################
params, params_covariance = optimize.curve_fit(test_func,  energies,  entries)
fit = test_func(energies, params[0], params[1])
# Goodness of fit
# residual sum of squares
ss_res = np.sum((entries - fit) ** 2)

# total sum of squares
ss_tot = np.sum((entries - np.mean(entries)) ** 2)

# r-squared
r2 = 1 - (ss_res / ss_tot)

print('m =', params[0])
print('b =', params[1])
print('r2 = '+str(r2))

plt.plot(energies, entries, 'ro-', linewidth=0.5, markersize=0.5)
# Label plot points with their values
for i, txt in enumerate(entries):
    if i>=9:
        plt.annotate(txt, (energies[i], entries[i]), xytext=(5,-5), textcoords='offset pixels', fontsize=4)
# Plot fitted function
plt.plot(energies, fit,
         label='Fitted function', linewidth=0.5, color='orange')

plt.title('Number of nCapture Events', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Number')
plt.savefig('images/nCaptureEntries.png', dpi=800, bbox_inches='tight')
plt.show()

# Try fit in log sace lower region

"""# Fitting only higher energies ################################################
etoplot = dfcount['energy'].tail(10)
ctoplot = dfcount['count'].tail(10)

# Fitting the count as a function of the energy
params, params_covariance = optimize.curve_fit(test_func,  etoplot,  ctoplot)
fit = test_func(etoplot, params[0], params[1])
# Goodness of fit
# residual sum of squares
ss_res = np.sum((ctoplot - fit) ** 2)

# total sum of squares
ss_tot = np.sum((ctoplot - np.mean(ctoplot)) ** 2)

# r-squared
r2 = 1 - (ss_res / ss_tot)

print('m =', params[0])
print('b =', params[1])
print('r2 = '+str(r2))

plt.plot(etoplot, ctoplot, 'ro-', linewidth=0.5, markersize=0.5)
# Label plot points with their values
for i, txt in enumerate(etoplot):
    plt.annotate(txt, (etoplot[i], ctoplot[i]), xytext=(5,-5), textcoords='offset pixels', fontsize=4)
# Plot fitted function
plt.plot(etoplot, fit,
         label='Fitted function', linewidth=0.5, color='orange')

plt.title('Number of nCapture Events', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Number')
plt.savefig('images/nCaptureEntries>=100MeV.png', dpi=800, bbox_inches='tight')
plt.show()"""