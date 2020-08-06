# %%
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata
import random
from scipy import stats
from multiprocessing import Pool
import seaborn as sns


# %%
# Read count and positions datasets
dfenergyncap = pd.read_csv('dfenergyncap.csv')
dfe = pd.read_csv('dfe_cylindrical.csv')

dfsim = pd.read_csv('sim_e100_n2000.csv')
df100 = dfe.loc[dfe['energy'] == 100]

# %%
# Plots rho and z
df100.plot.scatter('rho','z')
dfsim.plot.scatter('rho','z')

# %%
# Plots rho and t
df100.plot.scatter('rho','t')
dfsim.plot.scatter('rho','t')

# %%
# Plots z and t
df100.plot.scatter('z','t')
dfsim.plot.scatter('z','t')


# %%
# Scatter plot data 
# Calculate the point density
x = dfe['rho']
y = dfe['z']
xy = np.vstack([x,y])
z = stats.gaussian_kde(xy)(xy)

# Sort the points by density, so that the densest points are plotted last
idx = z.argsort()
x, y, z = x[idx], y[idx], z[idx]

fig, ax = plt.subplots()
ax.scatter(x, y, c=z, s=0.2, edgecolor='')
plt.show()

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
    df200,df300,df400,df500,df600,df700,df800,df900,df1000,df2000]

# Convert units to metres and microseconds
for df in dflist: 
    df[['t','x','y','z']] = df[['t','x','y','z']]/1000

# Create list of energies
energies = np.array([10,20,30,40,50,60,70,80,90,\
    100,200,300,400,500,600,700,800,900,1000,2000])

dfcouples = [(10,df10)\
        ,(20,df20)\
        ,(30,df30)\
        ,(40,df40)\
        ,(50,df50)\
        ,(60,df60)\
        ,(70,df70)\
        ,(80,df80)\
        ,(90,df90)\
        ,(100,df100)\
        ,(200,df200)\
        ,(300,df300)\
        ,(400,df400)\
        ,(500,df500)\
        ,(600,df600)\
        ,(700,df700)\
        ,(800,df800)\
        ,(900,df900)\
        ,(1000,df1000)\
        ,(2000,df2000)]

# %%
# Create cylindrical coordinates columns

for df in dflist:

    df['rho']=np.sqrt(dfe['x']**2+dfe['y']**2)
    df['theta']=np.arctan(dfe['y']/dfe['x'])
    df['theta'].fillna(0, inplace=True) # Replace NaN theta with 0
 

# %%
# Create column which counts number of nCapture in each event
for df in dflist:
    df['counteid'] = df.groupby('eventid')['eventid'].transform('count')

# %%

# Pairwise correlation matrix heatmap
# Shows rho, z and count are uncorrelated at given energies
i=0
for df in dflist:
    i+=1
    if i<11: 
        string = i*10 
    elif (i>=11 and i<20):
        string = (i-9)*100 
    elif i==20:
        string = i*100
    f, ax = plt.subplots(figsize=(4,3))
    corr = df[['t','rho','z','counteid']].corr(method='kendall')
    hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                    linewidths=.05)
    f.subplots_adjust(top=0.93)
    t= f.suptitle(f'Correlation Heatmap {string} MeV', fontsize=12)
    plt.savefig(f'images/correlation{string}.png', dpi=800, bbox_inches='tight')
    plt.show()
# %%
def plot_figures(figures, nrows = 1, ncols=1):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, ax1 = plt.subplots(ncols=ncols, nrows=nrows)
    gs1 = gridspec.GridSpec(nrows, ncols)
    gs1.update(wspace=0.025, hspace=0.025) # set the spacing between axes

    
    for ind,title in enumerate(figures):
        ax1 = plt.subplot(gs1[ind])
        ax1.imshow(figures[title])
        #axeslist.ravel()[ind].set_title(title)
        ax1.set_axis_off()
        ax1.set_aspect('equal')
        #plt.tight_layout() # optional
    plt.savefig('images/correlations_at_energies.png', dpi=800, bbox_inches='tight')


# %%
plt.figure(figsize = (4,4))
gs1 = gridspec.GridSpec(4, 4)
gs1.update(wspace=0.025, hspace=0.05) # set the spacing between axes. 

for i in range(16):
   # i = i + 1 # grid spec indexes from 0
    ax1 = plt.subplot(gs1[i])
    plt.axis('off')
    

plt.show()

# %%
img10 = mpimg.imread('images/correlation10.png')
img20 = mpimg.imread('images/correlation20.png')
img30 = mpimg.imread('images/correlation30.png')
img40 = mpimg.imread('images/correlation40.png')
img50 = mpimg.imread('images/correlation50.png')
img60 = mpimg.imread('images/correlation60.png')
img70 = mpimg.imread('images/correlation70.png')
img80 = mpimg.imread('images/correlation80.png')
img90 = mpimg.imread('images/correlation90.png')
img100 = mpimg.imread('images/correlation100.png')
img200 = mpimg.imread('images/correlation200.png')
img300 = mpimg.imread('images/correlation300.png')
img400 = mpimg.imread('images/correlation400.png')
img500 = mpimg.imread('images/correlation500.png')
img600 = mpimg.imread('images/correlation600.png')
img700 = mpimg.imread('images/correlation700.png')
img800 = mpimg.imread('images/correlation800.png')
img900 = mpimg.imread('images/correlation900.png')
img1000 = mpimg.imread('images/correlation1000.png')
img2000 = mpimg.imread('images/correlation2000.png')

titles = ['title1','title2','title3','title4','title5','title6','title7','title8','title9','title10',\
    'title11','title12','title13','title14','title15','title16','title17','title18','title19','title20']
imglist = [img10,img20,img30,img40,img50,img60,img70,img80,img90,img100,\
    img200,img300,img400,img500,img600,img700,img800,img900,img1000,img2000]
imgdict = dict(zip(titles,imglist))

# %%
titles = ['title1','title2','title3','title4','title5','title6']
imglist = [img10,img20,img30,img40,img50,img60]
imgdict = dict(zip(titles,imglist))

# %%
plot_figures(imgdict,nrows=5,ncols=4)

# %%
# Create dataframe with all energies
# Add energy column to all dataframes
for energy, dfc in dfcouples:
        dfc['energy']=energy

# Concatenate all dataframes
dfe = pd.concat(dflist, ignore_index=True)
dfe

# %%
f, ax = plt.subplots(figsize=(10, 6))
corr = dfe[['energy','t','rho','z','counteid']].corr(method='kendall')
hm = sns.heatmap(round(corr,2), annot=True, ax=ax, cmap="coolwarm",fmt='.2f',
                linewidths=.05)
f.subplots_adjust(top=0.93)
t= f.suptitle('Correlation Heatmap', fontsize=14)
plt.savefig('images/correlation100.png', dpi=800, bbox_inches='tight')
plt.show()
# So energy and z (and rho a little) are correllated overall, but not at individual energies
# rho, z, t, ncount are not related independently of energy, so we can sim them ALL independently
# Arguably z and counteid are a little correllated. 

# %%
fig = plt.figure()

ax1 = fig.add_subplot(211)
ax1.plot([(1, 2), (3, 4)], [(4, 3), (2, 3)])

ax2 = fig.add_subplot(212)
ax2.plot([(7, 2), (5, 3)], [(1, 6), (9, 5)])

plt.show()
plt.show()
# %%
df100.name = 'e100'
print(df100.name)
# %%

f, axarr = plt.subplots(2,2)
axarr[0,0].imshow(image_datas[0])
axarr[0,1].imshow(image_datas[1])
axarr[1,0].imshow(image_datas[2])
axarr[1,1].imshow(image_datas[3])