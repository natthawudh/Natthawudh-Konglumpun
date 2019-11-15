import numpy as np
class algorithm:
    def __init__(self, c = 1, eta = 0.002, max_iter = 1000, stop_citerier = 1e-4):
        self.eta = eta
        self.max_iter = max_iter
        self.stop_citerier = stop_citerier
    #Learning
    def fit(self,x,t):
        n = len(x[0])
        m = len(x)
        #initial variables
        self.w = np.zeros(n)
        self.b = 0.0
        y = np.dot(x, self.w) + self.b

        #iterative step
        for i in range(self.max_iter):
            dw = -2*np.matmul(x.T,(t-y))/m
            self.w += dw*self.eta
            db = -((t-y)*self.eta).sum()/m
            self.b += db
            h = np.dot(x, self.w) + self.b
            if np.sqrt(np.dot(dw,dw) + db**2) <= self.stop_citerier:
                break
    #prediction
    def predict(self, z):
        return np.dot(z, self.w) + self.b    