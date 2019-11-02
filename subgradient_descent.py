import numpy as np

class sgd:
    t = 0
    def __init__(self, loss='l1', L=1000):
        self.loss = loss
        self.L = L
        self.t = 0

    def step(self, x_t, z_t, y_t, eta=-1):
        if eta == -1:
            eta = np.sqrt(self.t)
        if self.loss == 'l1':
            loss_t = self.l1_loss(x_t, z_t, y_t)
            g_loss = self.g_l1_loss(x_t, z_t, y_t)
        elif this.loss == 'l2':
            loss_t = self.l2_loss(x_t, z_t, y_t)
            g_loss = self.g_l2_loss(x_t, z_t, y_t)
        
        x_new = self.greedy_projection(x_t - eta*g_loss)
        
        self.t = self.t + 1
        
        return loss_t, x_new

    def l1_loss(self, x_t, z_t, y_t):
        return np.absolute(y_t - np.max(np.multiply(x_t,z_t)))

    def l2_loss(self, x_t, z_t, y_t):
        raise NotImplementedError("todo.")

    # gradient of l1 loss
    def g_l1_loss(self, x_t, z_t, y_t):
        g_loss = y_t - np.max(np.multiply(x_t, z_t))
        g_loss = g_loss / np.absolute(y_t - np.max(np.multiply(x_t, z_t)))
        k = self.max_affine_hyperplane(x_t, z_t)
        
        v = np.zeros(z_t.shape)
        v[k] = z_t[k]
        g_loss = np.multiply(g_loss, v)
        return g_loss
    
    def max_affine_hyperplane(self, a, x):
        return np.argmax(np.multiply(a,x))
    # gradient of l2 loss
    def g_l2_loss(self, x_t, z_t, y_t):
        raise NotImplementedError("todo.")

    # greedy projection
    def greedy_projection(self, x):
        # projection based on the convex set of
        # max-affine functions where the norm is bounded
        for i in range(0,len(x)):
            if np.absolute(x[:,i]) > self.L:
                x[:,i] = self.L
        return x
