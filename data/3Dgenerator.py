import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random

min_val = -5
max_val = 5

X = np.random.uniform(min_val,max_val,size=(40,2))
x1 = np.arange(min_val,max_val,0.25)
x2 = np.arange(min_val,max_val,0.25)

# X = np.vstack([X_1, X_2])
# X = X.T
print(X.shape)

# add gaussian noise
mu = 0
sigma = 4
num_functions = 6
x1_v, x2_v = np.meshgrid(x1, x2, sparse=False, indexing='ij')

convex_functions = [np.sum(X**2, axis=1), np.sum(X**4, axis=1), np.sum(X**2+X**4, axis=1), 2*np.ones([X.shape[0],1]), np.sum(0.5*X**4,axis=1), np.sum(0.5*X**2, axis=1)]
mesh_grids = [np.multiply(x1_v, x1_v)+np.multiply(x2_v,x2_v), x1_v**4+x2_v**4, x1_v**2+x2_v**2+x1_v**4+x2_v**4, 2*np.ones(x1_v.shape), 0.5*(x1_v**4+x2_v**4), 0.5*(x1_v**2+x2_v**2)]
title_names = ["x^2", "x^4", "x^2 + x^4", "2", "0.5x^4", "0.5x^2"]

fig = plt.figure()
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

for i in range(0,num_functions):
    ax = fig.add_subplot(3,2,i+1, projection='3d')
    y = convex_functions[i]
    noise = np.random.normal(mu, sigma, size=(y.shape))
    y_noisy = np.add(y,noise)
    ax.scatter(X[:,0], X[:,1], y_noisy, c="r")

    surf = ax.plot_wireframe(x1_v,x2_v, mesh_grids[i], rcount=5,ccount=5)
    ax.set_title(title_names[i])
    ax.set_zlim(0,400)


plt.legend()
plt.show()

cm_colors = [cm.Purples, cm.Blues, cm.Greens, cm.Greys, cm.YlOrRd, cm.GnBu]
facecolors = ["purple", "blue", "green", "grey", "orange", "lightblue"]
fig = plt.figure()
ax = fig.add_subplot(1,1,1,projection='3d')
for i in range(0,num_functions):
    surf = ax.plot_surface(x1_v,x2_v, mesh_grids[i], cmap=cm_colors[i], facecolor=facecolors[i], linewidth=0, label=title_names[i])
    surf._facecolors2d=surf._facecolors3d
    surf._edgecolors2d=surf._edgecolors3d
plt.legend()
plt.show()


