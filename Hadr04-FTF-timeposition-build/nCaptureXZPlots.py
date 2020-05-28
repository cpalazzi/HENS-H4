# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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

fig, axes = plt.subplots(nrows=3, ncols=4)
fig.suptitle('nCapture XZ Positions', y=1.05)
xmin = -10000
xmax = 10000
ymin = -0
ymax = 20000

entries100 = len(df100)
axes[0,0].set_title("100 MeV", size=8)
axes[0,0].scatter(df100.x, df100.z, s=0.1)
axes[0,0].axes.set_xlim(xmin,xmax)
axes[0,0].axes.set_ylim(ymin,ymax)
# Turn off tick labels
axes[0,0].set_yticklabels([])
axes[0,0].set_xticklabels([])

entries200 = len(df200)
axes[0,1].set_title("200 MeV", size=8)
axes[0,1].scatter(df200.x, df200.z, s=0.1)
axes[0,1].axes.set_xlim(xmin,xmax)
axes[0,1].axes.set_ylim(ymin,ymax)
axes[0,1].set_yticklabels([])
axes[0,1].set_xticklabels([])

entries300 = len(df300)
axes[0,2].set_title("300 MeV", size=8)
axes[0,2].scatter(df300.x, df300.z, s=0.1)
axes[0,2].axes.set_xlim(xmin,xmax)
axes[0,2].axes.set_ylim(ymin,ymax)
axes[0,2].set_yticklabels([])
axes[0,2].set_xticklabels([])

entries400 = len(df400)
axes[0,3].set_title("400 MeV", size=8)
axes[0,3].scatter(df400.x, df400.z, s=0.1)
axes[0,3].axes.set_xlim(xmin,xmax)
axes[0,3].axes.set_ylim(ymin,ymax)
axes[0,3].set_yticklabels([])
axes[0,3].set_xticklabels([])

entries500 = len(df500)
axes[1,0].set_title("500 MeV", size=8)
axes[1,0].scatter(df500.x, df500.z, s=0.1)
axes[1,0].axes.set_xlim(xmin,xmax)
axes[1,0].axes.set_ylim(ymin,ymax)
axes[1,0].set_yticklabels([])
axes[1,0].set_xticklabels([])

entries600 = len(df600)
axes[1,1].set_title("600 MeV", size=8)
axes[1,1].scatter(df600.x, df600.z, s=0.1)
axes[1,1].axes.set_xlim(xmin,xmax)
axes[1,1].axes.set_ylim(ymin,ymax)
axes[1,1].set_yticklabels([])
axes[1,1].set_xticklabels([])

entries700 = len(df700)
axes[1,2].set_title("700 MeV", size=8)
axes[1,2].scatter(df700.x, df700.z, s=0.1)
axes[1,2].axes.set_xlim(xmin,xmax)
axes[1,2].axes.set_ylim(ymin,ymax)
axes[1,2].set_yticklabels([])
axes[1,2].set_xticklabels([])

entries800 = len(df800)
axes[1,3].set_title("800 MeV", size=8)
axes[1,3].scatter(df800.x, df800.z, s=0.1)
axes[1,3].axes.set_xlim(xmin,xmax)
axes[1,3].axes.set_ylim(ymin,ymax)
axes[1,3].set_yticklabels([])
axes[1,3].set_xticklabels([])

entries900 = len(df900)
axes[2,0].set_title("900 MeV", size=8)
axes[2,0].scatter(df900.x, df900.z, s=0.1)
axes[2,0].axes.set_xlim(xmin,xmax)
axes[2,0].axes.set_ylim(ymin,ymax)
axes[2,0].set_xlabel('x (mm)')
axes[2,0].set_ylabel('z (mm)')

axes[2,1].set_title("1000 MeV", size=8)
axes[2,1].scatter(df1000.x, df1000.z, s=0.1)
axes[2,1].axes.set_xlim(xmin,xmax)
axes[2,1].axes.set_ylim(ymin,ymax)
axes[2,1].set_yticklabels([])
axes[2,1].set_xticklabels([])

axes[2,2].set_visible(False)
axes[2,3].set_visible(False)

fig.tight_layout()
plt.savefig('nCaptureXZ.png', dpi=800, bbox_inches='tight')
plt.show()


