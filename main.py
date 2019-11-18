from subgradient_descent import sgd
import numpy as np
import matplotlib.pyplot as plt

k = 3
dims = 2
min_val = -1
max_val = 1
T = 200
DEBUG = True

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
    L = 1000

    # initialize h_t hypothesis
    # row corresponds to a given hyperplane
    # columns 0..dims are hyperplane slopes
    # last column are constants
    h_t = np.zeros((k,dims+1))
    
    X, y = generate_data()
    graph_1d(X,y)

    model = sgd(loss, L)

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