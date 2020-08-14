# %%
import pandas as pd
from matplotlib import pyplot as plt

#%%
df100 = pd.read_csv('timepositions100.csv', names=['t', 'x', 'y', 'z','eventid'])

df100.head()

# %%
plt.hist(df100['z'],bins=100)