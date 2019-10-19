class subgradient-descent:
    def __init__(loss='l1'):
        this.loss = loss
        this.t = 0

    def step(x_t,y_t,eta=-1):
        if eta == -1:
            eta = np.sqrt(this.t)
        if this.loss == 'l1':
            loss_t = l1_loss(x_t,y_t)
        return loss_t, x_new
        raise NotImplementedError("todo.")

    def l1_loss():

    # gradient of l1 loss
    def g_l1_loss(x_t):
        raise NotImplementedError("todo.")
    
    # gradient of l2 loss
    def g_l2_loss(x_t):
        raise NotImplementedError("todo.")
