import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import random
import pickle

def save_data(X,y,filename):
    f = open(filename, 'wb')
    pickle.dump(X,f)
    pickle.dump(y,f)
    f.close()

# graph for each convex function and corresponding data
def graph_function_subplots(num_functions, x1, x2, z, x1_mesh, x2_mesh, z_meshs, titles):
    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    for i in range(0,num_functions):
        ax = fig.add_subplot(3,2,i+1, projection='3d')

        # 3D scatter plot of generated data
        ax.scatter(x1, x2, z[i], c="r")

        # plot mesh in wire format so we can see both data and function surface
        ax.plot_wireframe(x1_mesh, x2_mesh, z_meshs[i], rcount=5,ccount=5)
        
        # specify title and z limits
        ax.set_title(titles[i])
        ax.set_zlim(0,400)

    # display all subplots
    plt.legend()
    plt.show()

# graph all meshes on 1 plot
def graph_all_functions(num_functions, x1_mesh, x2_mesh, z_meshes, titles):
    # plot all functions on one plot for comparison
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1,projection='3d')

    # specify colors for legend
    cm_colors = [cm.Purples, cm.Blues, cm.Greens, cm.Greys, cm.YlOrRd, cm.GnBu]
    facecolors = ["purple", "blue", "green", "grey", "orange", "lightblue"]

    for i in range(0,num_functions):
        surf = ax.plot_surface(x1_mesh,x2_mesh, z_meshes[i], cmap=cm_colors[i], facecolor=facecolors[i], linewidth=0, label=titles[i])
    
        # hack for plotting 3D legend
        surf._facecolors2d=surf._facecolors3d
        surf._edgecolors2d=surf._edgecolors3d

    # display plot
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # parameters
    num_functions = 6
    min_val = -5        # define data boundaries
    max_val = 5
    mu = 0              # gaussian noise mu, sigma
    sigma = 4

    # randomly generated data for training
    X = np.random.uniform(min_val,max_val,size=(40,2))
    # sequentially generated x1 and x2 for mesh plot
    x1_ordered = np.arange(min_val,max_val,0.25)
    x2_ordered = np.arange(min_val,max_val,0.25)
    x1_v, x2_v = np.meshgrid(x1_ordered, x2_ordered, sparse=False, indexing='ij')

    titles = ["x^2", "x^4", "x^2 + x^4", "2", "0.5x^4", "0.5x^2"]
    convex_functions = [np.sum(X**2, axis=1), np.sum(X**4, axis=1), np.sum(X**2+X**4, axis=1), 2*np.ones([X.shape[0],1]), np.sum(0.5*X**4,axis=1), np.sum(0.5*X**2, axis=1)]

    # equivalent formulation of convex functions for plotting mesh
    mesh_functions = [np.multiply(x1_v, x1_v)+np.multiply(x2_v,x2_v), x1_v**4+x2_v**4, x1_v**2+x2_v**2+x1_v**4+x2_v**4, 2*np.ones(x1_v.shape), 0.5*(x1_v**4+x2_v**4), 0.5*(x1_v**2+x2_v**2)]

    # add gaussian noise to each convex function
    noise = np.random.normal(mu, sigma, size=(convex_functions[0].shape))
    noisy_functions = [np.add(y, noise) for y in convex_functions]
    
    save_data(X,noisy_functions, "3d_data.pkl")
    graph_function_subplots(num_functions, X[:,0], X[:,1], convex_functions, x1_v, x2_v, mesh_functions, titles)
    graph_all_functions(num_functions, x1_v, x2_v, mesh_functions, titles)
