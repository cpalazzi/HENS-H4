#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:52:38 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Show number of entries
entries = []
for i in dflist:
    entries.append(len(i))
    
energies = [10,20,30,40,50,60,70,80,90\
            ,100,200,300,400,500,600,700,800,900,1000]

plt.plot(energies, entries, 'ro-', linewidth=0.5)
# Label plot points with their values
for i, txt in enumerate(entries):
    plt.annotate(txt, (energies[i], entries[i]), xytext=(5,-5), textcoords='offset pixels')
    
plt.title('Number of nCapture Events', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Number')
#plt.grid()
plt.savefig('nCaptureEntries_2.png', dpi=800, bbox_inches='tight')
plt.show()