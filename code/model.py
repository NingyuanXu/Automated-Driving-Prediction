import numpy as np
from tqdm import tqdm

from layers import Fc, Conv2d
from activations import Relu, Softmax

class Sequential:

    def __init__(self, layers = [], lr = 0.1, training = True, epochs = 10, batch_size = 500):
        self.layers = layers 
        self.optim = GD(lr)
        self.training = True
        self.epochs = epochs
        self.batch_size = batch_size

    
    def fit(self, X, y):
        epochs = self.epochs
        batch_size = self.batch_size
        n = X.shape[0]

        self.training = True
        # for i in tqdm(range(epochs)):
        for i in range(epochs):
            print('---------- epoch %s of %s----------------'%(i, epochs))
            X_batches, y_batches = findMin.gen_batch(X,y, self.batch_size)

            epoch_loss = 0
            # for X_batch, y_batch in zip(X_batches,y_batches):
            for X_batch, y_batch in tqdm(list(zip(X_batches,y_batches))):
                pred = self.forward(X_batch)
                first_grad = pred - y_batch.reshape(pred.shape) 
                self.backward(first_grad)
                self.optim.step(self.layers)

                epoch_loss += np.abs(first_grad).sum()/batch_size

                # print('curr_loss:', (first_grad**2).sum())
            
            print('loss: %s'%(epoch_loss))
    

    
    def predict(self, X):
        self.training = False
        return self.forward(X)
        


    def forward(self, X):
        n = X.shape[0]
        act = X 
        for l in self.layers:
            act = l.forward(act)
        
        if self.training == False:
            act = act.reshape(n,10)
            act = np.argmax(act, axis = -1)
        
        return act
    
    def backward(self, grad):
        g = grad 

        for l in reversed(self.layers):
            g = l.backward(g)



class GD:
    def __init__(self, lr):
        self.lr = lr 
    
    def step(self, layers):

        for l in layers:
            ws, g = l.get_weights(), l.get_grads()
        
            if None is ws[0]:
                continue
        
            (w,b) = ws 
            (gradw, gradb) = g
            w_new = w - self.lr*gradw
            b_new = b - self.lr*gradb
            l.update_weights(w_new, b_new)


def build_cnn(epochs = 10, batch_size = 512):
    conv1 = Conv2d(1,16,3,2,name = 1)
    conv2 = Conv2d(16,32,3,2,name = 2)
    conv3 = Conv2d(32,64,3,2,name = 3) 
    conv4 = Conv2d(64,10,2,1, name = 4)

    layers = [conv1, Relu(), conv2, Relu(), conv3, Relu(), conv4, Softmax()]

    network = Sequential(layers, epochs = epochs, batch_size=batch_size)

    return network

def build_neural_net(epochs = 10, batch_size = 512):

    fc1 = Fc(28**2,128)
    fc2 = Fc(128,64)
    fc3 = Fc(64,32)
    fc4 = Fc(32,10)

    layers = [fc1, Relu(),fc2,Relu(),fc3,Relu(),fc4,Softmax()]

    network = Sequential(layers, epochs = epochs, batch_size=batch_size)

    return network

