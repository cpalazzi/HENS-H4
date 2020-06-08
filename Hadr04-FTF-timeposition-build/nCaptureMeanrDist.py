#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:13:25 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('text', usetex=False)
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

dflist = [df10, df20, df30, df40, df50, df60, df70, df80, df90\
          , df100, df200, df300, df400, df500, df600, df700, df800, df900, df1000]

# Convert units to metres and microseconds
for df in dflist: 
    df[df.columns] = df[df.columns]/1000

# Create rdist columns
for i in dflist:
    i['rdist'] = np.sqrt(i['x']**2+i['y']**2+i['z']**2)

meanrdist = []
for i in dflist:
    meanrdist.append(round(i['rdist'].mean(), 3))

quantile05 = []
for i in dflist:
    quantile05.append(round(i['rdist'].quantile(0.05), 3))  
    
quantile32 = []
for i in dflist:
    quantile32.append(round(i['rdist'].quantile(0.32), 3))  
    
quantile95 = []
for i in dflist:
    quantile95.append(round(i['rdist'].quantile(0.95), 3)) 

quantile68 = []
for i in dflist:
    quantile68.append(round(i['rdist'].quantile(0.68), 3))
    
energies = [10,20,30,40,50,60,70,80,90\
            ,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

yerror = []
for i in dflist:
    yerror.append(i['rdist'].sem())
print(yerror)

def test_func(z, a, b):
    return a+b*np.log(z)

#def test_func(z, a, b):
#    return a+b*1/z


params, params_covariance = optimize.curve_fit(test_func,  energies,  meanrdist)

print('a =', params[0])
print('b =', params[1])

# Mean rdist by energy plot
plt.plot(energies, quantile95, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.95 quantile')
plt.plot(energies, quantile68, 'co--', linewidth=0.5, markersize=0.5,\
         label='0.68 quantile')

plt.plot(energies, meanrdist, 'bo-', linewidth=0.5, markersize=2, label='Mean r dist')
plt.errorbar(energies, meanrdist, yerr=yerror)

plt.plot(energies, quantile32, 'co--', linewidth=0.5, markersize=0.5,\
         label='0.32 quantile')
plt.plot(energies, quantile05, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.05 quantile')

plt.plot(energies, test_func(energies, params[0], params[1]),
         label='Fitted function')

# Label plot points with their values
for i, txt in enumerate(meanrdist):
    if i>=9:
        plt.annotate(txt, (energies[i], meanrdist[i]), xytext=(3,-10), textcoords='offset pixels')

plt.title('Mean r distance of nCapture', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Distance (m)')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc=(0.7,0.5))
plt.savefig('nCapturemeanrdist.png', dpi=800, bbox_inches='tight')
plt.show()

# rdist vs t scatterplots
fig, ax = plt.subplots(4, 5, sharex='col', sharey='row')
# axes are in a two-dimensional array, indexed by [row, col]
numplot = 0
for i in range(4):
    for j in range(5):
        if numplot<len(dflist):
            if numplot<10:
                ax[i, j].text(5, 1600, str((numplot+1)*10)+' MeV'+'\n'+str(len(dflist[numplot]['rdist']))+' entries', fontsize=4, ha='center')
            elif numplot>=10:
                ax[i, j].text(5, 1600, str((numplot-8)*100)+' MeV'+'\n'+str(len(dflist[numplot]['rdist']))+' entries', fontsize=4, ha='center')
            ax[i, j].scatter(dflist[numplot]['rdist'], dflist[numplot]['t'], s=0.1, edgecolors='none')
            ax[i, j].set_xlim(left=0, right=14)
            ax[i, j].set_ylim(bottom=0, top=2100)
            numplot+=1

fig.suptitle('r Distance and Time of nCapture')
fig.text(0.5, 0.01, 'r Distance (m)', ha='center', va='center')
fig.text(0.01, 0.5, 'Time (microsec)', ha='center', va='center', rotation='vertical')

plt.savefig('nCapturerdist.png', dpi=800, bbox_inches='tight')
