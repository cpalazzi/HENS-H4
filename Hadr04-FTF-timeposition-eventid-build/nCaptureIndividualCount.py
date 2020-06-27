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
from scipy.optimize import minimize
from scipy.special import factorial
from scipy import stats
# %%
# Read and combine datasets
df1 = pd.read_csv('timepositions1.csv', names=['t', 'x', 'y', 'z','eventid'])
df2 = pd.read_csv('timepositions2.csv', names=['t', 'x', 'y', 'z','eventid'])
df3 = pd.read_csv('timepositions3.csv', names=['t', 'x', 'y', 'z','eventid'])
df4 = pd.read_csv('timepositions4.csv', names=['t', 'x', 'y', 'z','eventid'])
df5 = pd.read_csv('timepositions5.csv', names=['t', 'x', 'y', 'z','eventid'])
df6 = pd.read_csv('timepositions6.csv', names=['t', 'x', 'y', 'z','eventid'])
df7 = pd.read_csv('timepositions7.csv', names=['t', 'x', 'y', 'z','eventid'])
df8 = pd.read_csv('timepositions8.csv', names=['t', 'x', 'y', 'z','eventid'])
df9 = pd.read_csv('timepositions9.csv', names=['t', 'x', 'y', 'z','eventid'])
df10 = pd.read_csv('timepositions10.csv', names=['t', 'x', 'y', 'z','eventid'])
df11 = pd.read_csv('timepositions11.csv', names=['t', 'x', 'y', 'z','eventid'])
df12 = pd.read_csv('timepositions12.csv', names=['t', 'x', 'y', 'z','eventid'])
df13 = pd.read_csv('timepositions13.csv', names=['t', 'x', 'y', 'z','eventid'])
df14 = pd.read_csv('timepositions14.csv', names=['t', 'x', 'y', 'z','eventid'])
df15 = pd.read_csv('timepositions15.csv', names=['t', 'x', 'y', 'z','eventid'])
df16 = pd.read_csv('timepositions16.csv', names=['t', 'x', 'y', 'z','eventid'])
df17 = pd.read_csv('timepositions17.csv', names=['t', 'x', 'y', 'z','eventid'])
df18 = pd.read_csv('timepositions18.csv', names=['t', 'x', 'y', 'z','eventid'])
df19 = pd.read_csv('timepositions19.csv', names=['t', 'x', 'y', 'z','eventid'])
df20 = pd.read_csv('timepositions20.csv', names=['t', 'x', 'y', 'z','eventid'])
df30 = pd.read_csv('timepositions30.csv', names=['t', 'x', 'y', 'z','eventid'])
df40 = pd.read_csv('timepositions40.csv', names=['t', 'x', 'y', 'z','eventid'])
df50 = pd.read_csv('timepositions50.csv', names=['t', 'x', 'y', 'z','eventid'])
df60 = pd.read_csv('timepositions60.csv', names=['t', 'x', 'y', 'z','eventid'])
df70 = pd.read_csv('timepositions70.csv', names=['t', 'x', 'y', 'z','eventid'])
df80 = pd.read_csv('timepositions80.csv', names=['t', 'x', 'y', 'z','eventid'])
df90 = pd.read_csv('timepositions90.csv', names=['t', 'x', 'y', 'z','eventid'])
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
df2000 = pd.read_csv('timepositions2000.csv', names=['t', 'x', 'y', 'z','eventid'])

dflist = [df10,df20,df30,df40,df50,df60,df70,df80,df90,df100,\
    df100,df200,df300,df400,df500,df600,df700,df800,df900,df1000,df2000]

# Convert units to metres and microseconds
for df in dflist: 
    df[['t','x','y','z']] = df[['t','x','y','z']]/1000

# Create list of energies
energies = np.array([10,20,30,40,50,60,70,80,90,\
    100,200,300,400,500,600,700,800,900,1000,2000])
 
# %%
numevents = 2000

# Generate dfcount dataframes
#10
dictcount10 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df10[df10['eventid'] == i]) for i in range(numevents)],
        }
dfcount10 = pd.DataFrame(dictcount10, columns = ['eventid','ncapcount'])
#20
dictcount20 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df20[df20['eventid'] == i]) for i in range(numevents)],
        }
dfcount20 = pd.DataFrame(dictcount20, columns = ['eventid','ncapcount'])
#30
dictcount30 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df30[df30['eventid'] == i]) for i in range(numevents)],
        }
dfcount30 = pd.DataFrame(dictcount30, columns = ['eventid','ncapcount'])
#40
dictcount40 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df40[df40['eventid'] == i]) for i in range(numevents)],
        }
dfcount40 = pd.DataFrame(dictcount40, columns = ['eventid','ncapcount'])
#50
dictcount50 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df50[df50['eventid'] == i]) for i in range(numevents)],
        }
dfcount50 = pd.DataFrame(dictcount50, columns = ['eventid','ncapcount'])
#60
dictcount60 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df60[df60['eventid'] == i]) for i in range(numevents)],
        }
dfcount60 = pd.DataFrame(dictcount60, columns = ['eventid','ncapcount'])
#70
dictcount70 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df70[df70['eventid'] == i]) for i in range(numevents)],
        }
dfcount70 = pd.DataFrame(dictcount70, columns = ['eventid','ncapcount'])
#80
dictcount80 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df80[df80['eventid'] == i]) for i in range(numevents)],
        }
dfcount80 = pd.DataFrame(dictcount80, columns = ['eventid','ncapcount'])
#90
dictcount90 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df90[df90['eventid'] == i]) for i in range(numevents)],
        }
dfcount90 = pd.DataFrame(dictcount90, columns = ['eventid','ncapcount'])
#100
dictcount100 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df100[df100['eventid'] == i]) for i in range(numevents)],
        }
dfcount100 = pd.DataFrame(dictcount100, columns = ['eventid','ncapcount'])
#200
dictcount200 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df200[df200['eventid'] == i]) for i in range(numevents)],
        }
dfcount200 = pd.DataFrame(dictcount200, columns = ['eventid','ncapcount'])
#300
dictcount300 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df300[df300['eventid'] == i]) for i in range(numevents)],
        }
dfcount300 = pd.DataFrame(dictcount300, columns = ['eventid','ncapcount'])
#400
dictcount400 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df400[df400['eventid'] == i]) for i in range(numevents)],
        }
dfcount400 = pd.DataFrame(dictcount400, columns = ['eventid','ncapcount'])
#500
dictcount500 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df500[df500['eventid'] == i]) for i in range(numevents)],
        }
dfcount500 = pd.DataFrame(dictcount500, columns = ['eventid','ncapcount'])
#600
dictcount600 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df600[df600['eventid'] == i]) for i in range(numevents)],
        }
dfcount600 = pd.DataFrame(dictcount600, columns = ['eventid','ncapcount'])
#700
dictcount700 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df700[df700['eventid'] == i]) for i in range(numevents)],
        }
dfcount700 = pd.DataFrame(dictcount700, columns = ['eventid','ncapcount'])
#800
dictcount800 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df800[df800['eventid'] == i]) for i in range(numevents)],
        }
dfcount800 = pd.DataFrame(dictcount800, columns = ['eventid','ncapcount'])
#900
dictcount900 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df900[df900['eventid'] == i]) for i in range(numevents)],
        }
dfcount900 = pd.DataFrame(dictcount900, columns = ['eventid','ncapcount'])
#1000
dictcount1000 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df1000[df1000['eventid'] == i]) for i in range(numevents)],
        }
dfcount1000 = pd.DataFrame(dictcount1000, columns = ['eventid','ncapcount'])
#2000
dictcount2000 = {'eventid':  list(range(0,numevents)),
        'ncapcount': [len(df2000[df2000['eventid'] == i]) for i in range(numevents)],
        }
dfcount2000 = pd.DataFrame(dictcount2000, columns = ['eventid','ncapcount'])

dfcountlist = [dfcount10,dfcount20,dfcount30,dfcount40,dfcount50,dfcount60,dfcount70,dfcount80,dfcount90,\
    dfcount100,dfcount200,dfcount300,dfcount400,dfcount500,dfcount600,dfcount700,dfcount800,dfcount900,dfcount1000,dfcount2000]

# %%
# Plot single histogram
plt.hist(dfcount100['ncapcount'],bins=np.arange(12)-0.5)
plt.xticks(range(12))
plt.show()

# %%
# ncapcount histograms
fig, ax = plt.subplots(4, 5, sharex='col', sharey='row')
# axes are in a two-dimensional array, indexed by [row, col]
numplot = 0
for i in range(4):
    for j in range(5):
        if numplot<len(dfcountlist):
            if numplot<10:
                ax[i, j].text(10, 1200, str((numplot+1)*10)+' MeV'+'\n'+str(dfcountlist[numplot]['ncapcount'].sum())+' entries', fontsize=4, ha='center')
            if numplot>=10 and numplot<19:
                ax[i, j].text(10, 1200, str((numplot-8)*100)+' MeV'+'\n'+str(dfcountlist[numplot]['ncapcount'].sum())+' entries', fontsize=4, ha='center')
            if numplot==19:
                ax[i, j].text(10, 1200, str((numplot+1)*100)+' MeV'+'\n'+str(dfcountlist[numplot]['ncapcount'].sum())+' entries', fontsize=4, ha='center') 
            ax[i, j].hist(dfcountlist[numplot]['ncapcount'], np.arange(20)-0.5)
            ax[i, j].set_xlim(left=-2, right=20)
            ax[i, j].set_ylim(bottom=0, top=2000)
            ax[i, j].set_xticks(range(0,20,5))

            numplot+=1

fig.suptitle('Number of events with multiple nCapture counts')
fig.text(0.5, 0.01, 'Number of nCapture', ha='center', va='center')
fig.text(0.01, 0.5, 'Number of events', ha='center', va='center', rotation='vertical')

plt.savefig('images/nCapturecountmulti.png', dpi=800, bbox_inches='tight')
plt.show()



# %%
# Create df with energies and ncap counts for individual events. 
# Should have 40000 rows
dfcountcouples = [(10,dfcount10)\
        ,(20,dfcount20)\
        ,(30,dfcount30)\
        ,(40,dfcount40)\
        ,(50,dfcount50)\
        ,(60,dfcount60)\
        ,(70,dfcount70)\
        ,(80,dfcount80)\
        ,(90,dfcount90)\
        ,(100,dfcount100)\
        ,(200,dfcount200)\
        ,(300,dfcount300)\
        ,(400,dfcount400)\
        ,(500,dfcount500)\
        ,(600,dfcount600)\
        ,(700,dfcount700)\
        ,(800,dfcount800)\
        ,(900,dfcount900)\
        ,(1000,dfcount1000)\
        ,(2000,dfcount2000)]

# %%
# Add energy column to all dataframes
for energy, dfc in dfcountcouples:
        dfc['energy']=energy

print(dfcount2000.head())

# %%
dfenergyncap = pd.concat(dfcountlist, ignore_index=True)

print(len(dfenergyncap))

# %%
plt.scatter(dfenergyncap['energy'],dfenergyncap['ncapcount'], s=0.5)
plt.xlabel('Energy (MeV)')
plt.ylabel('Number of nCap in event')
plt.savefig('images/E_nCap_scatter.png', dpi=800, bbox_inches='tight')
plt.show()

# %%
plt.hist2d(dfenergyncap['energy'], dfenergyncap['ncapcount'], cmap=plt.cm.jet, bins=50)
plt.xlabel('Energy (MeV)')
plt.ylabel('Number of nCap in event')
plt.colorbar().set_label('Number of events')
plt.savefig('images/E_nCap_2dhisto.png', dpi=800, bbox_inches='tight')
plt.show()

# # %%
# def poisson(k, lamb):
#     """poisson pdf, parameter lamb is the fit parameter"""
#     return (lamb**k/factorial(k)) * np.exp(-lamb)

# def negative_log_likelihood(params, data):
#     ''' better alternative using scipy '''
#     return -stats.poisson.logpmf(data, params[0]).sum()

# data10 = dfenergyncap['ncapcount'].where(dfenergyncap['energy']==10)

# # minimize the negative log-Likelihood

# result = minimize(negative_log_likelihood,  # function to minimize
#                   x0=np.ones(1),            # start value
#                   args=(data10,),             # additional arguments for function
#                   method='Powell',          # minimization method, see docs
#                   )
# # result is a scipy optimize result object, the fit parameters 
# # are stored in result.x
# print(result)


# %%
# Create test function
def poisson(k, lamb, scale):
    return scale*(lamb**k/factorial(k))*np.exp(-lamb)


# Fitting the count as a function of the energy ############################
params, params_covariance = optimize.curve_fit(poisson,  dfcount100['ncapcount'])
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

# %%
df100bar = df100bar/2000
df100bar.plot(kind='bar')

# %%
c = dfenergyncap.groupby(['energy','ncapcount'], as_index=False).size().to_frame()
c.head()
c['name'] = 'size'
# %%
