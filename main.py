from subgradient_descent import sgd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import imageio

def load_data(filename, func_number):
    f = open(filename, 'rb')
    X = pickle.load(f)
    y = pickle.load(f)
    T, dims = X.shape
    return X,y[func_number], T, dims

def generate_data(T, dims):
    np.random.seed(0)
    x = np.random.uniform(min_val, max_val, size = (T,dims) )
    y = np.sum(np.abs(0.5*x**2), axis=1)
    return x,y, T, dims

def graph_1d(x,y):
    if dims != 1:
        print("Could not graph non-planar data")
        return

    plt.plot(x,y, '*')
    plt.savefig("data_graph.png")
    plt.close()

def graph_hypothesis(x,y,h,t):
    if dims != 1:
        print("Could not graph non-planar data")
        return
    
    for i in range(0,h.shape[0]):
        y_vals = h[i,0]*x + h[i,-1]
        plt.plot(x,y_vals)

    plt.plot(x[:t],y[:t],"*")
    plt.ylim([-5,25])
    plt.title("Timestep: "+str(t))
    fname = "graphs/max_affine_"+str(t)+".png"
    plt.savefig(fname)
    plt.close()
    return fname

def graph_loss(losses,T):
    numbers = range(1,T)
    plt.plot(numbers,losses)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Iteration vs. Loss")
    plt.savefig("loss.png")
    plt.close()

def make_gif(filenames):
    images = []

    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave('hypothesis.gif', images)

def train_model(X, y, T, dims, graph_frequency=100):
    # init h_t hypothesis
    # row corresponds to a given hyperplane
    # columns 0..dims are hyperplane slopes
    # last column are constants
    h_t = np.zeros((k,dims+1))
    
    # init model
    model = sgd(loss, L)
    losses = []
    filenames = []

    for t in range(1,T):
        print(t)
        x_t = X[t,:]
        y_t = y[t]

        loss_t, h_t = model.step(h_t, x_t, y_t)
        losses.append(loss_t)
        if t%graph_frequency == 0:
            fname=graph_hypothesis(X,y,h_t,t)
            filenames.append(fname)
    
    make_gif(filenames)
    return losses
    
    
if __name__ == "__main__":
    loss = "l1"
    L = 1
    k = 10
    min_val = -5
    max_val = 5
    DEBUG = False
    func_number = 0 # quadratic x**2
    T = 2000
    dims = 1
    # X, y, T, dims = load_data("data/2d_data.pkl",0)
    X, y, T, dims = generate_data(T, dims)
    graph_1d(X,y)

    losses = train_model(X, y, T, dims, graph_frequency=25)
    # graph_hypothesis(X,y,h_t,T)
    graph_loss(losses, T)