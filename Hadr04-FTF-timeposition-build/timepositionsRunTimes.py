#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 13:39:10 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dfRunTimes = pd.read_csv('timepositionRunTimes.csv')

# Show number of entries
totalMinutes = dfRunTimes['TotalTime (mins)']
energies = [100,200,300,400,500,600,700,800,900,1000]
#plt.scatter(energies, totalMinutes, s=20, c='g')
plt.plot(energies, totalMinutes, 'go-', linewidth=0.5)
# Label plot points with their values
for i, txt in enumerate(totalMinutes):
    plt.annotate(txt, (energies[i], totalMinutes[i]), xytext=(-15,5), textcoords='offset pixels')
    
plt.title('Run Times for HENS-H4 for Primary Action of 2000 Neutrons', y=1.05)
plt.ylim(-5,310)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Run Time (mins)')
plt.tight_layout()
plt.savefig('nCaptureRunTimes.png', dpi=800, bbox_inches='tight')
plt.show()