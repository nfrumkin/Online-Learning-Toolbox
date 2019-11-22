import numpy as np
from cvxopt import matrix, solvers

class ftrl:
    # ==== Notation ====
    # h = hypothesis
    # z_t = data
    # y_t = true label
    # hypothesis max-affine is given by argmax(h*z_t)
    # gnd truth max-affine is y_t
    
    t = 1
    g_losses = []
    def __init__(self, loss='l1', L=1000):
        self.loss = loss
        self.L = L
        self.t = 1

    def step(self, h_t, z_t, y_t, eta=-1):
        if eta == -1:
            eta = 1/np.sqrt(self.t)
        
        # multiplicative weight update
        new_exp = np.e(-1*eta*(np.sum(z_t))
        cumulative_sum = cumulative_sum + new_exp
        h_new = new_exp/(cumulative_sum)

        raise NotImplementedError("specify loss")
        self.t = self.t + 1
        
        return loss_t, h_new

    
    def max_affine(self, h_t, z):
        vals = np.dot(h_t[:,:-1],z)+h_t[:,-1]
        return np.max(vals)

    def max_affine_number(self, h_t, z):
        vals = np.dot(h_t[:,:-1],z)+h_t[:,-1]
        return np.argmax(vals)

