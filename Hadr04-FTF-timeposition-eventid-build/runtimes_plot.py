# %%
import matplotlib.pyplot as plt
import pandas as pd

# %%
energyruntimes = pd.read_csv('energyruntimes.csv',names=['energy','runtimes'])
# Convert s to hours
energyruntimes['runtimes']=energyruntimes['runtimes']/(60*60)

# %%
plt.plot(energyruntimes['energy'],energyruntimes['runtimes'], '-o', markersize=3)
plt.xlabel('Energy (MeV)')
plt.ylabel('Run time (h)')
plt.title('HENS run times for 2000 events')
plt.savefig('images/runtimesplot.png', dpi=800, bbox_inches='tight')
plt.show()


# %%
