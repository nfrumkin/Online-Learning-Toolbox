from subgradient_descent import sgd
import numpy as np
import matplotlib.pyplot as plt
import pickle


def load_data(filename, func_number):
    f = open(filename, 'rb')
    X = pickle.load(f)
    y = pickle.load(f)
    T, dims = X.shape
    return X,y[func_number], T, dims

def generate_data():
    np.random.seed(0)
    x = np.random.uniform(min_val, max_val, size = (T,dims) )
    y = np.sum(x**2, axis=1)
    return x,y  

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

    plt.plot(x,y,"*")
    plt.savefig("graphs/max_affine_" + str(t) + ".png")
    plt.close()

def graph_loss(losses):
    numbers = range(1,T)
    plt.plot(numbers,losses)
    plt.savefig("loss.png")
    plt.close()

if __name__ == "__main__":
    loss = "l1"
    losses = []
    L = 100
    k = 3
    min_val = -1
    max_val = 1
    DEBUG = False
    func_number = 0 # quadratic x**2
    
    X, y, T, dims = load_data("data/2d_data.pkl",0)
    graph_1d(X,y)

    # initialize h_t hypothesis
    # row corresponds to a given hyperplane
    # columns 0..dims are hyperplane slopes
    # last column are constants
    h_t = np.zeros((k,dims+1))

    model = sgd(loss, L)
    print("T: ", T)
    for t in range(1,T):
        print(t)
        x_t = X[t,:]
        y_t = y[t]

        loss_t, h_t = model.step(h_t, x_t, y_t)
        losses.append(loss_t)
        if DEBUG:
            # graph_hypothesis(X,y,h_t,t)
            print("x_t: ", x_t, ", y_t: ", y_t)
            print("loss: ", loss_t)
            print("h_t: ", h_t)

    # if DEBUG:
    graph_hypothesis(X,y,h_t,T)
    graph_loss(losses)