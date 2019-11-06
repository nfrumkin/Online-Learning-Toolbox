from subgradient_descent import sgd
import numpy as np
import matplotlib.pyplot as plt

k = 3
dims = 1
min_val = -1
max_val = 1
T = 2
DEBUG = True

def generate_data():
    np.random.seed(0)
    x = np.random.uniform(min_val, max_val, size = (100,dims) )
    y = np.sum(x**2, axis=1)
    return x,y     

def print_loss(losses):
    numbers = range(0,T)
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
    model = sgd(loss, L)

    for t in range(0,T):
        x_t = X[t,:]
        y_t = y[t]

        loss_t, h_t = model.step(h_t, x_t, y_t)
        losses.append(loss_t)
        if DEBUG:
            print("x_t: ", x_t, ", y_t: ", y_t)
            print("loss: ", loss_t)
            print("h_t: ", h_t)

    if DEBUG:
        print_loss(losses)

