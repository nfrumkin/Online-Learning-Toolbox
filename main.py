from subgradient_descent import sgd

if __name__ == "__main__":
    loss = "l1"
    losses = []
    T = 100
    L = 1000
    model = sgd(loss, L)
    for t in range(0,T):
        loss_t, x_t = sgd.step(x_t, z_t, y_t)
        losses.append(loss_t)

