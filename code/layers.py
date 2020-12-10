# layer APIs inspired by torch and tensorflow
# Really barebone implementation of what's needed for finals 
# full of really bad coding practices

# support only fully connected and cnn
import numpy as np


scale = 0.01

class Layer:
    def __init__(self):
        self.w = None
        self.b = None 

        self.gradw = None 
        self.gradb = None
    
    def get_weights(self):
        return self.w, self.b
    
    def get_grads(self):
        return self.gradw, self.gradb

class Fc(Layer):
    def __init__(self, in_shape:int, out_shape:int):
        super().__init__()
        self.w = scale * np.random.rand(out_shape, in_shape)
        self.b = scale * np.random.rand(1, out_shape)

        self.inp = None
        self.gradw = None 
        self.gradb = None
    

    def forward(self, X):
        self.inp = X 
        return X@self.w.T + self.b 
    
    #act_grad = gradients from activation
    #compute gradients
    def backward(self, act_grad): 
        n = len(self.inp)

        self.gradw = (act_grad.T@self.inp)/n
        self.gradb = np.sum(act_grad, axis = 0).reshape(self.b.shape)/n

        return act_grad@self.w # 
    

    
    def update_weights(self, w_new, b_new):
        self.w = w_new 
        self.b = b_new 


class Conv2d(Layer):

    # k_size = kernel_size
    # c_size = channel size = in_shape
    # f_size = filter_size = out_shape
    # stride is always 1
    # no padding, life is already complicated enough
    # not gonna do maxpooling
    def __init__(self, in_c, out_c,k_size, stride = 1, name = 'no_name'):
        super().__init__()
        self.w = scale*np.random.rand(k_size, k_size, in_c, out_c)
        self.b = scale*np.random.rand(out_c)
        self.stride = stride

        self.gradw = None
        self.gradb = None

        self.name = 'Conv_' + str(name)
    

    def forward(self, X):
        self.inp = X

        n, h_in, w_in, c = X.shape 

        k, k, in_c, out_c = self.w.shape

        #calc output dim
        w_out = (w_in - k)//self.stride + 1 # okay, working only with squared images for test
        h_out = (h_in - k)//self.stride + 1

        output = np.zeros((n, h_out, w_out, out_c))

        for i in range(h_out):
            for j in range(w_out):
                # get index for image patch
                top_left = i*self.stride
                top_right = i*self.stride + k 
                bottom_left = j*self.stride 
                bottom_right = j*self.stride + k 

                vol_x = X[:, top_left:top_right, bottom_left:bottom_right,:]

                for curr_c in range(out_c):
                    output[:,i,j, curr_c] = (vol_x*self.w[:,:,:,curr_c]).sum(axis = (1,2,3)) + self.b[curr_c]

         
        return output 
    
    def backward(self, act_grad):
        # gradient formula intuition from: https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
       
        n, h_out, w_out, c_out = act_grad.shape
        n, h_in, w_in, c_in = self.inp.shape 

        k, k, _, w_c = self.w.shape

        output = np.zeros(self.inp.shape)
        

        self.gradb = act_grad.sum(axis = (0,1,2))/n  #quickest way to get to shape of b; check math later
        self.gradw = np.zeros(self.w.shape)

        output1 = np.zeros(self.inp.shape)
        dw = np.zeros(self.w.shape)

        
        

        for i in range(h_out):
            for j in range(w_out):
                top_left = i*self.stride
                top_right = i*self.stride + k 
                bottom_left = j*self.stride 
                bottom_right = j*self.stride + k

                vol_act = act_grad[:, i:i+1, j:j+1, :]#act_grad[:, i, j, :]
                vol_x = self.inp[:, top_left:top_right, bottom_left:bottom_right,:]


                ## dy/dx
                for inp_c in  range(c_in):
                    output[:,top_left:top_right, bottom_left:bottom_right, inp_c] += (vol_act*self.w[:,:,inp_c,:]).sum(axis = -1)
                

                #dy/dw
                for curr_c in range(w_c):
                    self.gradw[:,:,:,curr_c] += (vol_x*vol_act[:,:,:,curr_c][:,None]).sum(axis = 0)
                  
        self.gradw = self.gradw/n


        return output


    def update_weights(self, w_new, b_new):
        self.w = w_new 
        self.b = b_new








        














        



    