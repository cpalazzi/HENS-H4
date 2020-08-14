#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:00:42 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('text', usetex=False)
from scipy import optimize

df100 = pd.read_csv('timepositions100.csv', names=['t', 'x', 'y', 'z','eventid'])
df200 = pd.read_csv('timepositions200.csv', names=['t', 'x', 'y', 'z','eventid'])
df300 = pd.read_csv('timepositions300.csv', names=['t', 'x', 'y', 'z','eventid'])
df400 = pd.read_csv('timepositions400.csv', names=['t', 'x', 'y', 'z','eventid'])
df500 = pd.read_csv('timepositions500.csv', names=['t', 'x', 'y', 'z','eventid'])
df600 = pd.read_csv('timepositions600.csv', names=['t', 'x', 'y', 'z','eventid'])
df700 = pd.read_csv('timepositions700.csv', names=['t', 'x', 'y', 'z','eventid'])
df800 = pd.read_csv('timepositions800.csv', names=['t', 'x', 'y', 'z','eventid'])
df900 = pd.read_csv('timepositions900.csv', names=['t', 'x', 'y', 'z','eventid'])
df1000 = pd.read_csv('timepositions1000.csv', names=['t', 'x', 'y', 'z','eventid'])

dflist = [df100, df200, df300, df400, df500, df600, df700, df800, df900, df1000]

meanydist = []
for i in dflist:
    meanydist.append(round((((np.sqrt(i['y'])**2)).mean())/1000, 3))

quantile05 = []
for i in dflist:
    quantile05.append(round(((np.sqrt(i['y'])**2)).quantile(0.05)/1000, 3))  
    
quantile32 = []
for i in dflist:
    quantile32.append(round(((np.sqrt(i['y'])**2)).quantile(0.32)/1000, 3))  
    
quantile95 = []
for i in dflist:
    quantile95.append(round(((np.sqrt(i['y'])**2)).quantile(0.95)/1000, 3)) 

quantile68 = []
for i in dflist:
    quantile68.append(round(((np.sqrt(i['y'])**2)).quantile(0.68)/1000, 3))
    
print(meanydist)
print(quantile68)
    
energies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

## Convert lists to arrays
#meanydist = np.array(meanydist)
#quantile05 = np.array(quantile05)
#quantile95 = np.array(quantile95)
#energies = np.array(energies)

#def test_func(z, a, b):
#    return a+b*np.log(z)

def test_func(z, a, b):
    return a+b*1/z


params, params_covariance = optimize.curve_fit(test_func,  energies,  meanydist)

print('a =', params[0])
print('b =', params[1])


plt.plot(energies, quantile95, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.95 quantile')
plt.plot(energies, quantile68, 'ro--', linewidth=0.5, markersize=0.5,\
         label='0.68 quantile')
plt.plot(energies, meanydist, 'bo-', linewidth=0.5, markersize=2, label='Mean y dist')
plt.plot(energies, quantile32, 'co--', linewidth=0.5, markersize=0.5,\
         label='0.32 quantile')
plt.plot(energies, quantile05, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.05 quantile')

plt.plot(energies, test_func(energies, params[0], params[1]),
         label='Fitted function')

# Label plot points with their values
for i, txt in enumerate(meanydist):
    plt.annotate(txt, (energies[i], meanydist[i]), xytext=(3,-10), textcoords='offset pixels')
    
plt.title('Mean y distance of nCapture', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Distance (m)')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc=(0.7,0.5))
plt.savefig('nCapturemeanydist.png', dpi=800, bbox_inches='tight')
plt.show()
    
    