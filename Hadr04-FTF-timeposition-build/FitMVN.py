#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 15:41:46 2020

@author: carlopalazzi
"""

import pandas as pd
import numpy as np
from statsmodels.graphics.gofplots import qqplot
from matplotlib import pyplot

from pydoc import help  # can type in the python console `help(name of function)` to get the documentation
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy import stats
from scipy.optimize import curve_fit
from IPython.display import display, HTML

import pingouin as pg

def func(X, n0, tau, mu1, sigma1, a1, mu2, sigma2, a2, mu3, sigma3, a3, b):
    t, x, y, z = X
    return n0*np.exp(-t/tau) + a1*np.exp(((x-mu1)**2)/sigma1) + \
            a1*np.exp(((x-mu1)**2)/sigma1) + a2*np.exp(((x-mu2)**2)/sigma2) + \
            a3*np.exp(((x-mu3)**2)/sigma3) + b

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


# initial guesses for parameters
p0 = 1, 0.2, df100['x'].mean(), df100['x'].var(), 1, \
    df100['y'].mean(), df100['y'].var(), 1, \
    df100['z'].mean(), df100['z'].var(), 1, 1
print(curve_fit(func, X, df100, p0))

df100['x'].hist()

pd.plotting.scatter_matrix(df100, diagonal="kde")
plt.tight_layout()
plt.show()

g = sns.lmplot("x", "z", df100, hue="t", fit_reg=False);
g._legend.remove()

cov100 = df100.cov()
mean100 = df100.mean()
std100 = df100.std
print(cov100)
print("\n")
print(mean100)

cov1000 = df1000.cov()
mean1000 = df1000.mean()
print(cov1000)
print("\n")
print(mean1000)

# q-q plot
qqplot(df100['t'], line='s', dist=stats.laplace)
pyplot.show()
qqplot(df100['x'], line='s', dist=stats.laplace)
pyplot.show()
qqplot(df100['z'], line='s', dist=stats.laplace)
pyplot.show()

# multivariate normality test
normal, p = pg.multivariate_normality(df100[['x','y','z']], alpha=.05)
print(normal, round(p, 3))
# interpret
alpha = 0.05
if p > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')
   
# univariate normality test
stat, p1 = stats.normaltest(df100['x'])
print('Statistics=%.3f, p1=%.3f' % (stat, p1))
# interpret
alpha = 0.05
if p1 > alpha:
	print('Sample looks Gaussian (fail to reject H0)')
else:
	print('Sample does not look Gaussian (reject H0)')

