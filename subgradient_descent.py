import numpy as np

class sgd:
    # ==== Notation ====
    # h = hypothesis
    # z_t = data
    # y_t = true label
    # hypothesis max-affine is given by argmax(h*z_t)
    # gnd truth max-affine is y_t
    
    t = 0
    def __init__(self, loss='l1', L=1000):
        self.loss = loss
        self.L = L
        self.t = 0

    def step(self, h_t, z_t, y_t, eta=-1):
        if eta == -1:
            eta = np.sqrt(self.t)
        if self.loss == 'l1':
            loss_t = self.l1_loss(h_t, z_t, y_t)
            g_loss = self.g_l1_loss(h_t, z_t, y_t)
        elif this.loss == 'l2':
            loss_t = self.l2_loss(h_t, z_t, y_t)
            g_loss = self.g_l2_loss(h_t, z_t, y_t)
        
        h_new = self.greedy_projection(h_t - eta*g_loss)
        
        self.t = self.t + 1
        
        return loss_t, h_new

    def l1_loss(self, h_t, z_t, y_t):
        return np.absolute(y_t - self.max_affine(h_t,z_t))

    def l2_loss(self, h_t, z_t, y_t):
        raise NotImplementedError("todo.")

    # gradient of l1 loss
    def g_l1_loss(self, h_t, z_t, y_t):

        k = self.max_affine_number(h_t, z_t)
        max_val = self.max_affine(h_t, z_t)

        g_loss = np.zeros(h_t.shape)

        abs_val = np.absolute(y_t - max_val)
        sign = abs_val/(y_t - max_val)

        # assign b_k to be the opposite of the abs sign
        g_loss[k,-1] = (-1)*sign
        
        # assign a_k to be the sign multiplied by z_t
        g_loss[k,:-1] = sign*z_t

        return g_loss
    
    def max_affine(self, h_t, z):
        # import pdb
        # pdb.set_trace()
        vals = np.dot(h_t[:,:-1],z)+h_t[:,-1]
        return np.max(vals)

    def max_affine_number(self, h, z):
        return np.argmax(np.multiply(h, z))
    # gradient of l2 loss
    def g_l2_loss(self, h_t, z_t, y_t):
        raise NotImplementedError("todo.")

    # greedy projection
    def greedy_projection(self, h):
        # projection based on the convex set of
        # max-affine functions where the norm is bounded
        for row in range(0,h.shape[0]):
            if np.sum(np.absolute(h[row,:-1])) > self.L:
                raise NotImplementedError("todo: apply greedy projection onto L-norm ball")
        return h
