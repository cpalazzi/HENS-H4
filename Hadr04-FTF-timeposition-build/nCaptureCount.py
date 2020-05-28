#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:52:38 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

entries100 = len(df100)
entries200 = len(df200)
entries300 = len(df300)
entries400 = len(df400)
entries500 = len(df500)
entries600 = len(df600)
entries700 = len(df700)
entries800 = len(df800)
entries900 = len(df900)
entries1000 = len(df1000)

# Show number of entries
entries = [entries100
           , entries200
           , entries300
           , entries400
           , entries500
           , entries600
           , entries700
           , entries800
           , entries900
           , entries1000]
energies = [100,200,300,400,500,600,700,800,900,1000]
plt.plot(energies, entries, 'ro-', linewidth=0.5)
# Label plot points with their values
for i, txt in enumerate(entries):
    plt.annotate(txt, (energies[i], entries[i]), xytext=(5,-5), textcoords='offset pixels')
    
plt.title('Number of nCapture Events', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Number')
#plt.grid()
plt.savefig('nCaptureEntries.png', dpi=800, bbox_inches='tight')
plt.show()