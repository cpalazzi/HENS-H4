#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:15:26 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('text', usetex=False)
from scipy import optimize
from brokenaxes import brokenaxes


df10 = pd.read_csv('timepositions10.csv', names=['t', 'x', 'y', 'z'])
df10['v']=np.sqrt(df10['x']**2+df10['y']**2+df10['z']**2)/df10['t']
df20 = pd.read_csv('timepositions20.csv', names=['t', 'x', 'y', 'z'])
df20['v']=np.sqrt(df20['x']**2+df20['y']**2+df20['z']**2)/df20['t']
df30 = pd.read_csv('timepositions30.csv', names=['t', 'x', 'y', 'z'])
df30['v']=np.sqrt(df30['x']**2+df30['y']**2+df30['z']**2)/df30['t']
df40 = pd.read_csv('timepositions40.csv', names=['t', 'x', 'y', 'z'])
df40['v']=np.sqrt(df40['x']**2+df40['y']**2+df40['z']**2)/df40['t']
df50 = pd.read_csv('timepositions50.csv', names=['t', 'x', 'y', 'z'])
df50['v']=np.sqrt(df50['x']**2+df50['y']**2+df50['z']**2)/df50['t']
df60 = pd.read_csv('timepositions60.csv', names=['t', 'x', 'y', 'z'])
df60['v']=np.sqrt(df60['x']**2+df60['y']**2+df60['z']**2)/df60['t']
df70 = pd.read_csv('timepositions70.csv', names=['t', 'x', 'y', 'z'])
df70['v']=np.sqrt(df70['x']**2+df70['y']**2+df70['z']**2)/df70['t']
df80 = pd.read_csv('timepositions80.csv', names=['t', 'x', 'y', 'z'])
df80['v']=np.sqrt(df80['x']**2+df80['y']**2+df80['z']**2)/df80['t']
df90 = pd.read_csv('timepositions90.csv', names=['t', 'x', 'y', 'z'])
df90['v']=np.sqrt(df90['x']**2+df90['y']**2+df90['z']**2)/df90['t']
df100 = pd.read_csv('timepositions100.csv', names=['t', 'x', 'y', 'z'])
df100['v']=np.sqrt(df100['x']**2+df100['y']**2+df100['z']**2)/df100['t']
df200 = pd.read_csv('timepositions200.csv', names=['t', 'x', 'y', 'z'])
df200['v']=np.sqrt(df200['x']**2+df200['y']**2+df200['z']**2)/df200['t']
df300 = pd.read_csv('timepositions300.csv', names=['t', 'x', 'y', 'z'])
df300['v']=np.sqrt(df300['x']**2+df300['y']**2+df300['z']**2)/df300['t']
df400 = pd.read_csv('timepositions400.csv', names=['t', 'x', 'y', 'z'])
df400['v']=np.sqrt(df400['x']**2+df400['y']**2+df400['z']**2)/df400['t']
df500 = pd.read_csv('timepositions500.csv', names=['t', 'x', 'y', 'z'])
df500['v']=np.sqrt(df500['x']**2+df500['y']**2+df500['z']**2)/df500['t']
df600 = pd.read_csv('timepositions600.csv', names=['t', 'x', 'y', 'z'])
df600['v']=np.sqrt(df600['x']**2+df600['y']**2+df600['z']**2)/df600['t']
df700 = pd.read_csv('timepositions700.csv', names=['t', 'x', 'y', 'z'])
df700['v']=np.sqrt(df700['x']**2+df700['y']**2+df700['z']**2)/df700['t']
df800 = pd.read_csv('timepositions800.csv', names=['t', 'x', 'y', 'z'])
df800['v']=np.sqrt(df800['x']**2+df800['y']**2+df800['z']**2)/df800['t']
df900 = pd.read_csv('timepositions900.csv', names=['t', 'x', 'y', 'z'])
df900['v']=np.sqrt(df900['x']**2+df900['y']**2+df900['z']**2)/df900['t']
df1000 = pd.read_csv('timepositions1000.csv', names=['t', 'x', 'y', 'z'])
df1000['v']=np.sqrt(df1000['x']**2+df1000['y']**2+df1000['z']**2)/df1000['t']

dflist = [df10, df20, df30, df40, df50, df60, df70, df80, df90\
          , df100, df200, df300, df400, df500, df600, df700, df800, df900, df1000]

meanv = []
for i in dflist:
    meanv.append(round(i['v'].mean(), 3))

quantile05 = []
for i in dflist:
    quantile05.append(round(i['v'].quantile(0.05), 3))  
    
quantile32 = []
for i in dflist:
    quantile32.append(round(i['v'].quantile(0.32), 3))  
    
quantile95 = []
for i in dflist:
    quantile95.append(round(i['v'].quantile(0.95), 3)) 

quantile68 = []
for i in dflist:
    quantile68.append(round(i['v'].quantile(0.68), 3))
    
energies = [10,20,30,40,50,60,70,80,90,100\
            , 200, 300, 400, 500, 600, 700, 800, 900, 1000]

## Convert lists to arrays
#meanv = np.array(meanv)
#quantile05 = np.array(quantile05)
#quantile95 = np.array(quantile95)
#energies = np.array(energies)

def test_func(z, a, b):
    return a+b*np.log(z)

#def test_func(z, a, b):
#    return a+b*1/z


params, params_covariance = optimize.curve_fit(test_func,  energies,  meanv)

print('a =', params[0])
print('b =', params[1])


plt.plot(energies, quantile95, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.95 quantile')
plt.plot(energies, quantile68, 'co--', linewidth=0.5, markersize=0.5,\
         label='0.68 quantile')
plt.plot(energies, meanv, 'bo-', linewidth=0.5, markersize=2, label='Mean v')
plt.plot(energies, quantile32, 'co--', linewidth=0.5, markersize=0.5,\
         label='0.32 quantile')
plt.plot(energies, quantile05, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.05 quantile')

plt.plot(energies, test_func(energies, params[0], params[1]),
         label='Fitted function')

# Label plot points with their values
for i, txt in enumerate(meanv):
    if i>=9:
        plt.annotate(txt, (energies[i], meanv[i]), xytext=(3,-10), textcoords='offset pixels')
    
plt.title('Mean neutron radial speed', y=1.05)
plt.xlabel('Neutron Energy (MeV)')
plt.ylabel('Speed (m/microsec)')
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.legend(loc=(0.7,0.46))
plt.savefig('nCapturemeanv.png', dpi=800, bbox_inches='tight')
plt.show()

# Plotting histograms
df10['v'].hist(bins=100)
plt.show()

x = df10['v']
bax = brokenaxes(ylims=((0, 10), (5740, 5750)))
bax.hist(x, histtype='bar', bins=200)
plt.title('Mean neutron radial speed for 10MeV', y=1.05)
bax.set_xlabel('Mean Radial Speed (m/microseconds)')
bax.set_ylabel(' ')
plt.savefig('v10histo.png', dpi=800, bbox_inches='tight')
plt.show()


np.log10(df300['v'].copy()).hist()
plt.title('Mean neutron radial speed for 300MeV', y=1.05)
plt.xlabel('log(Mean Radial Speed (m/microseconds))')
plt.ylabel('Number of events')
plt.savefig('v300loghisto.png', dpi=800, bbox_inches='tight')
plt.show()
    
