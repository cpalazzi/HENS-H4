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
fig.suptitle('nCapture XY Positions', y=1.05)
xmin = -10000
xmax = 10000

axes[0,0].set_title("100 MeV", size=8)
axes[0,0].scatter(df100.x, df100.y, s=0.1)
axes[0,0].axes.set_xlim(xmin,xmax)
axes[0,0].axes.set_ylim(xmin,xmax)
# Turn off tick labels
axes[0,0].set_yticklabels([])
axes[0,0].set_xticklabels([])


axes[0,1].set_title("200 MeV", size=8)
axes[0,1].scatter(df200.x, df200.y, s=0.1)
axes[0,1].axes.set_xlim(xmin,xmax)
axes[0,1].axes.set_ylim(xmin,xmax)
axes[0,1].set_yticklabels([])
axes[0,1].set_xticklabels([])

axes[0,2].set_title("300 MeV", size=8)
axes[0,2].scatter(df300.x, df300.y, s=0.1)
axes[0,2].axes.set_xlim(xmin,xmax)
axes[0,2].axes.set_ylim(xmin,xmax)
axes[0,2].set_yticklabels([])
axes[0,2].set_xticklabels([])

axes[0,3].set_title("400 MeV", size=8)
axes[0,3].scatter(df400.x, df400.y, s=0.1)
axes[0,3].axes.set_xlim(xmin,xmax)
axes[0,3].axes.set_ylim(xmin,xmax)
axes[0,3].set_yticklabels([])
axes[0,3].set_xticklabels([])

axes[1,0].set_title("500 MeV", size=8)
axes[1,0].scatter(df500.x, df500.y, s=0.1)
axes[1,0].axes.set_xlim(xmin,xmax)
axes[1,0].axes.set_ylim(xmin,xmax)
axes[1,0].set_yticklabels([])
axes[1,0].set_xticklabels([])

axes[1,1].set_title("600 MeV", size=8)
axes[1,1].scatter(df600.x, df600.y, s=0.1)
axes[1,1].axes.set_xlim(xmin,xmax)
axes[1,1].axes.set_ylim(xmin,xmax)
axes[1,1].set_yticklabels([])
axes[1,1].set_xticklabels([])

axes[1,2].set_title("700 MeV", size=8)
axes[1,2].scatter(df700.x, df700.y, s=0.1)
axes[1,2].axes.set_xlim(xmin,xmax)
axes[1,2].axes.set_ylim(xmin,xmax)
axes[1,2].set_yticklabels([])
axes[1,2].set_xticklabels([])

axes[1,3].set_title("800 MeV", size=8)
axes[1,3].scatter(df800.x, df800.y, s=0.1)
axes[1,3].axes.set_xlim(xmin,xmax)
axes[1,3].axes.set_ylim(xmin,xmax)
axes[1,3].set_yticklabels([])
axes[1,3].set_xticklabels([])

axes[2,0].set_title("900 MeV", size=8)
axes[2,0].scatter(df900.x, df900.y, s=0.1)
axes[2,0].axes.set_xlim(xmin,xmax)
axes[2,0].axes.set_ylim(xmin,xmax)
axes[2,0].set_xlabel('x (mm)')
axes[2,0].set_ylabel('y (mm)')

axes[2,1].set_title("1000 MeV", size=8)
axes[2,1].scatter(df1000.x, df1000.y, s=0.1)
axes[2,1].axes.set_xlim(xmin,xmax)
axes[2,1].axes.set_ylim(xmin,xmax)
axes[2,1].set_yticklabels([])
axes[2,1].set_xticklabels([])

axes[2,2].set_visible(False)
axes[2,3].set_visible(False)

fig.tight_layout()
plt.savefig('nCaptureXY.png', dpi=800, bbox_inches='tight')
plt.show()