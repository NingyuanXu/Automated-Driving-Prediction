import numpy as np
from layers import Layer
import warnings

class Relu(Layer):
    def __init__(self):
        super().__init__()
        self.act = None 
    
    def forward(self,X):
        self.act = np.maximum(0,X)

        return self.act 
    
    def backward(self, grad_act):

        grad = grad_act.copy()
        grad[self.act <= 0] = 0 
        return grad
    

class Softmax(Layer):
    def __init__(self):
        super().__init__()
        self.act = None 
    
    def forward(self, X):
        X_shift = X - X.max(axis = -1, keepdims = True)
        norm = np.exp(X_shift)
        denom = norm.sum(axis=-1, keepdims = True)
        self.act = norm/denom

        try:
            warnings.warn(Warning())
        except Warning:
            print('caugth a warning')
        
        return self.act
    
    def backward(self, grad_act):
        # derivative = output - y, grad_act already is that
        return grad_act
