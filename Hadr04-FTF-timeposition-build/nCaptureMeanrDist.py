#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 29 11:08:25 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from matplotlib import rc
#rc('text', usetex=False)
from scipy import optimize
from scipy import stats

df10 = pd.read_csv('timepositions10.csv', names=['t', 'x', 'y', 'z'])
df10['r']=np.sqrt(df10['x']**2+df10['y']**2+df10['z']**2)
df20 = pd.read_csv('timepositions20.csv', names=['t', 'x', 'y', 'z'])
df20['r']=np.sqrt(df20['x']**2+df20['y']**2+df20['z']**2)
df30 = pd.read_csv('timepositions30.csv', names=['t', 'x', 'y', 'z'])
df30['r']=np.sqrt(df30['x']**2+df30['y']**2+df30['z']**2)
df40 = pd.read_csv('timepositions40.csv', names=['t', 'x', 'y', 'z'])
df40['r']=np.sqrt(df40['x']**2+df40['y']**2+df40['z']**2)
df50 = pd.read_csv('timepositions50.csv', names=['t', 'x', 'y', 'z'])
df50['r']=np.sqrt(df50['x']**2+df50['y']**2+df50['z']**2)
df60 = pd.read_csv('timepositions60.csv', names=['t', 'x', 'y', 'z'])
df60['r']=np.sqrt(df60['x']**2+df60['y']**2+df60['z']**2)
df70 = pd.read_csv('timepositions70.csv', names=['t', 'x', 'y', 'z'])
df70['r']=np.sqrt(df70['x']**2+df70['y']**2+df70['z']**2)
df80 = pd.read_csv('timepositions80.csv', names=['t', 'x', 'y', 'z'])
df80['r']=np.sqrt(df80['x']**2+df80['y']**2+df80['z']**2)
df90 = pd.read_csv('timepositions90.csv', names=['t', 'x', 'y', 'z'])
df90['r']=np.sqrt(df90['x']**2+df90['y']**2+df90['z']**2)
df100 = pd.read_csv('timepositions100.csv', names=['t', 'x', 'y', 'z'])
df100['r']=np.sqrt(df100['x']**2+df100['y']**2+df100['z']**2)
df200 = pd.read_csv('timepositions200.csv', names=['t', 'x', 'y', 'z'])
df200['r']=np.sqrt(df200['x']**2+df200['y']**2+df200['z']**2)
df300 = pd.read_csv('timepositions300.csv', names=['t', 'x', 'y', 'z'])
df300['r']=np.sqrt(df300['x']**2+df300['y']**2+df300['z']**2)
df400 = pd.read_csv('timepositions400.csv', names=['t', 'x', 'y', 'z'])
df400['r']=np.sqrt(df400['x']**2+df400['y']**2+df400['z']**2)
df500 = pd.read_csv('timepositions500.csv', names=['t', 'x', 'y', 'z'])
df500['r']=np.sqrt(df500['x']**2+df500['y']**2+df500['z']**2)
df600 = pd.read_csv('timepositions600.csv', names=['t', 'x', 'y', 'z'])
df600['r']=np.sqrt(df600['x']**2+df600['y']**2+df600['z']**2)
df700 = pd.read_csv('timepositions700.csv', names=['t', 'x', 'y', 'z'])
df700['r']=np.sqrt(df700['x']**2+df700['y']**2+df700['z']**2)
df800 = pd.read_csv('timepositions800.csv', names=['t', 'x', 'y', 'z'])
df800['r']=np.sqrt(df800['x']**2+df800['y']**2+df800['z']**2)
df900 = pd.read_csv('timepositions900.csv', names=['t', 'x', 'y', 'z'])
df900['r']=np.sqrt(df900['x']**2+df900['y']**2+df900['z']**2)
df1000 = pd.read_csv('timepositions1000.csv', names=['t', 'x', 'y', 'z'])
df1000['r']=np.sqrt(df1000['x']**2+df1000['y']**2+df1000['z']**2)

dflist = [df10,df20,df30,df40,df50,df60,df70,df80,df90\
          ,df100, df200, df300, df400, df500, df600, df700, df800, df900, df1000]

meanrdist = []
for i in dflist:
    meanrdist.append(round((((np.sqrt(i['r'])**2)).mean())/1000, 3))
      
yerror = []
for i in dflist:
    yerror.append((np.sqrt(i['r'])**2).sem())
print(yerror)

ystd = []
for i in dflist:
    ystd.append((np.sqrt(i['r'])**2).std()/np.sqrt(len(i)))


quantile05 = []
for i in dflist:
    quantile05.append(round(i['r'].quantile(0.05)/1000, 3))  
    
quantile32 = []
for i in dflist:
    quantile32.append(round(i['r'].quantile(0.32)/1000, 3))  
    
quantile95 = []
for i in dflist:
    quantile95.append(round(i['r'].quantile(0.95)/1000, 3)) 

quantile68 = []
for i in dflist:
    quantile68.append(round(i['r'].quantile(0.68)/1000, 3))
    
energies = [10,20,30,40,50,60,70,80,90,100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

countspmin = pd.DataFrame(np.array([[1,0],[2,2],[3,5],[4,7]]))
print(countspmin.sem())

#print(np.sqrt(1830**2-1600**2))

## Convert lists to arrays
#meanrdist = np.array(meanrdist)
#quantile05 = np.array(quantile05)
#quantile95 = np.array(quantile95)
#energies = np.array(energies)

def test_func(z, a, b):
    return a+b*np.log(z)

#def test_func(z, a, b):
#    return a+b*1/z


params, params_covariance = optimize.curve_fit(test_func,  energies,  meanrdist)

print('a =', params[0])
print('b =', params[1])


plt.plot(energies, quantile95, 'bo--', linewidth=0.5, markersize=0.5,\
         label='0.95 quantile')
plt.plot(energies, quantile68, 'co--', linewidth=0.5, markersize=0.5,\
         label='0.68 quantile')

plt.plot(energies, meanrdist, 'bo-', linewidth=0.5, markersize=2, label='Mean r dist')
plt.errorbar(energies, meanrdist, yerr=ystd)

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
plt.legend(loc=(0.7,0.46))
plt.savefig('nCapturemeanrdist.png', dpi=800, bbox_inches='tight')
plt.show()
    
    