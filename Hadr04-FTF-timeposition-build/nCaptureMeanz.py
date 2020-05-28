#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 14:29:46 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import scipy

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

dflist = [df100, df200, df300, df400, df500, df600, df700, df800, df900, df1000]

meanz = []
for i in dflist:
    meanz.append(round((i['z'].mean())/1000, 3))

mostlyabove = []
for i in dflist:
    mostlyabove.append(round(i['z'].quantile(0.05)/1000, 3))   
    
mostlybelow = []
for i in dflist:
    mostlybelow.append(round(i['z'].quantile(0.95)/1000, 3)) 
    
energies = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

## Convert lists to arrays
#meanz = np.array(meanz)
#mostlyabove = np.array(mostlyabove)
#mostlybelow = np.array(mostlybelow)
#energies = np.array(energies)

#def test_func(z, a, b):
#    return a+b*np.log(z)

def test_func(z, a, b):
    return a+b*z**(1/3)

params, params_covariance = scipy.optimize.curve_fit(test_func,  energies,  meanz)



plt.plot(energies, mostlybelow, 'ro--', linewidth=0.5, markersize=0.5,\
         label='0.95 quantile')
plt.plot(energies, meanz, 'bo-', linewidth=0.5, markersize=2, label='Mean z')
plt.plot(energies, mostlyabove, 'go--', linewidth=0.5, markersize=0.5,\
         label='0.05 quantile')

plt.plot(energies, test_func(energies, params[0], params[1]),
         label='Fitted function')

# Label plot points with their values
for i, txt in enumerate(meanz):
    plt.annotate(txt, (energies[i], meanz[i]), xytext=(3,-10), textcoords='offset pixels')
    
plt.title('Mean z of nCapture', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('z (m)')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend()
plt.savefig('nCapturemeanz.png', dpi=800, bbox_inches='tight')
plt.show()
    
    