import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from math import sqrt
import numpy as np
from scipy.interpolate import bisplrep, bisplev

x1 = np.array([0, 1, 10, 20, 30, 50, 60, 70, 100, 300, 500, 700, 1000])
x2 = np.array([0, 1, 10, 30, 50, 100, 300, 500, 700, 800, 850, 900, 950, 1000])
#x1 = np.array([0, 1, 10, 50, 100, 300, 500, 700, 1000])
#x2 = np.array([0, 1, 10, 50, 100, 300, 500, 700, 1000])
x3 = np.array([0, 1, 10, 50, 100, 300, 500, 700, 1000])

# 100 GeV (0 from 100m onwards)
y1 = np.array([100 for _ in range(13)])
z1 = np.array([100.097, 99.5066,93.8384,87.1023,81.1629,68.6166,62.4541,
			   56.8929, 40.3017, 0, 0, 0, 0])
z1_std = np.array([0.018903,0.523505,3.84514,7.54244,7.24474,9.7405,10.0773,
                   9.76382,8.82479, 0, 0, 0, 0])
#z1 = np.array([100.097, 99.5066,93.8384,68.6166,40.3017,0,0,0,0])
#z1_std = np.array([0.018903,0.523505,3.84514,9.7405,8.82479,0,0,0,0])
z1_err = z1_std/sqrt(1000)

# 500 GeV (0 for 1km)
y2 = np.array([500 for _ in range(14)])
z2 = np.array([500.098,499.321,488.59,469.602,450.318,399.662,233.957,110.366,
               24.2043,0,0,0,0,0])
z2_std = np.array([0.0184366,1.32562,31.1142,29.8654,42.781,62.018,64.203,
                   45.9802,17.1419,0,0,0,0,0])
#z2 = np.array([500.098,499.321,488.59,450.318,399.662,233.957,110.366,
#               24.2043,0])
#z2_std = np.array([0.0184366,1.32562,31.1142,42.781,62.018,64.203,
#                   45.9802,17.1419,0])
z2_err = z2_std/sqrt(1000)

# 1 TeV
y3 = np.array([1000 for _ in range(9)])
z3 = np.array([1000.09,998.603,983.87,922.564,852.08,606.532,405.89,255.09,
               96.7455])
z3_std = np.array([0.133969,8.56668,47.2303,89.6812,122.636,154.945,144.65,
                   116.142,60.1205])
z3_err = z3_std/sqrt(1000)

# concatenate everything together to form one array for each dimension
x_tot = np.concatenate((x1, x2, x3))
y_tot = np.concatenate((y1, y2, y3))
z_tot = np.concatenate((z1, z2, z3))
z_tot_err = np.concatenate((z1_err, z2_err, z3_err))

# weights for the 2d spline fit
weight = np.array([1/i if i != 0 else 10 for i in z_tot_err])

print(len(x_tot), len(y_tot), len(z_tot), len(z_tot_err),len(weight))
print(z_tot_err,weight)
# spline fit
bispl = bisplrep(x_tot, y_tot, z_tot, w=weight)

print(bispl)

x_tot_fit = np.linspace(0, 1000, 100)
y_tot_fit = np.linspace(100, 1000, 100)
z_tot_fit = bisplev(x_tot_fit, y_tot_fit, bispl)

print(len(z_tot_fit))

'''
plt.errorbar(x1,z1,yerr=z1_err,fmt='x',label='100 GeV mu- source')
plt.errorbar(x2,z2,yerr=z2_err,fmt='x',label='500 GeV mu- source')
plt.errorbar(x3,z3,yerr=z3_err,fmt='x',label='1 TeV mu- source')
'''
# plot 2d spline fits
plt.plot(x_tot_fit, z_tot_fit,color="0.5")

# plot straight line y=0 to see if the fit went below 0
x_line = np.linspace(0, 1000, 3)
y_line = [0 for _ in x_line]
plt.plot(x_line, y_line, '-.')

plt.xlabel('No. of Neutron Captured')
plt.ylabel('No. of Event')
plt.title('Energy of Muon When Entering Water')

#plt.yscale('log')
plt.legend()
plt.show()



