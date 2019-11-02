from subgradient_descent import sgd
import numpy as np

k = 5
def generate_data():
    x = np.random.uniform(-1, 0, size = (100,k) )
    y = np.sum(x**2, axis=1)
    return x,y

if __name__ == "__main__":
    loss = "l1"
    losses = []
    T = 100
    L = 1000

    x_t = np.zeros( (1,k) )
    z, y = generate_data()
    model = sgd(loss, L)

    for t in range(0,T):
        z_t = z[t,:]
        y_t = y[t]
        loss_t, x_t = model.step(x_t, z_t, y_t)
        losses.append(loss_t)
        print("x_t: ", x_t)

