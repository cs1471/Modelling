""" 
DNN is a little python module to demonstrate 
the basic elements of a deep neural network 
in action.
"""
import numpy as np
import matplotlib.pyplot as plt

from pointwise_activations import func_list
from loss_functions import loss_list

class Data(object):
    """
    Data class takes care of the minibatch management

    There are two optional parameters for Data objects.
    -- batch_size: if left as None will use the entire data set for 
       each pass.
    -- shuffle: wich defaults to true will shuffle data points to feed 
       the network at each iteration. When a epoch has been reached, 
       data is reshuffled.
    """
    batch_iter = None 
    data_size = None
    batch_idx = None
    crt_idx = None
    n_batches = None
    shuffle = None
    
    def __init__(self, data, batch_size=None, shuffle=True):
        self.data = data
        self.data_size = data.shape[0]
        self.shuffle = shuffle
        if batch_size is None:
            self.batch_size = self.data_size
            self.batch_iter = None
            self.batch_idx = None
            self.shuffle = False
        else:
            assert self.data_size >= batch_size, 'batch_size exeeds number of data samples'
            self.batch_size = batch_size
            self.batch_iter = 0
        self.n_batches = self.data_size / self.batch_size
        leftovers = self.data_size % self.batch_size
                
    def getBatch(self):
        if self.batch_iter is None:
            return self.data
  
        if self.batch_iter == 0:
            ## shuffle data at the begining of every epoch
            if self.shuffle is True:
                self.batch_idx = np.random.permutation(self.data_size) 
            else:
                self.batch_idx = np.arange(self.data_size)
            self.batch_idx = np.reshape(self.batch_idx, (self.n_batches, -1))
        self.crt_idx = self.batch_idx[self.batch_iter, :]
        batch = self.data[self.crt_idx, ::]
        self.batch_iter += 1
        if self.batch_iter == self.n_batches:
            self.batch_iter = 0
        return batch

    def getDataAsIn(self, ref_data_object=None):
        """
        This is an auxiliary method with the purpose of selecting
        the corresponding data-target pairs. 
        """
        if ref_data_object.crt_idx is not None:
            return self.data[ref_data_object.crt_idx, ::]
        else:
            return self.data
    

class Layer(object):
    """
    Layer class implements a uniform composition of affine map followed by 
    point-wise nonlinearity.
    
    Input: 
    -- n_in: number of inputs
    -- n_out: number of outputs (number of units)
    -- activation: point-wise nonlinearity. 
       --logistic, 
       --tanh, 
       --relu, 
       --abs,
       --square, 
       --halfsquare
    """

    X0 = None
    X1 = None
    Z = None
    D0 = None
    D1 = None
    W = None
    b = None
    Delta_W = None
    Delta_b = None
    g = None
    g_prime = None
    n_in = None
    n_out = None

    def __init__(self, n_in, n_out, activation):
        assert n_in is not None and n_out is not None, "layer must have valid input output sizes"
        self.n_in = n_in
        self.n_out = n_out
        self.W = np.random.normal(size=(self.n_in, self.n_out)) / np.sqrt(self.n_in)
        self.b = np.zeros((1, self.n_out))
        self.Delta_W = np.zeros((self.n_in, self.n_out))
        self.Delta_b = np.zeros((1, self.n_out))
        self.g = func_list[activation][0]
        self.g_prime = func_list[activation][1]
    
        
    def forward(self):
        self.Z = np.dot(self.X0, self.W) + self.b
        self.X1 = self.g(self.Z)
    
    def backward(self):
        if self.D1 is None:
            self.G = self.g_prime(self.Z)
        else:
            self.G = np.multiply(self.D1, self.g_prime(self.Z))
        self.D0 = np.dot(self.G, self.W.transpose())

    def updateParam(self, solver_func):
        self.Delta_W, self.Delta_b = solver_func(self)
        self.W += self.Delta_W
        self.b += self.Delta_b

class Net(object):
    """
    Net is a container for all the Layer objects that form a network

    It manages the forward,backward passes through the networks as well
    as the parametr update calls. Note that Net objects are independent of
    the cost function employed for their training
    """
    
    n_layer = 0
    layers = []
    Xout = None
    def __init__(self):
        self.n_layer = 0
        self.layers = []
        self.Xout = None
    
    def addLayer(self, n_in=None, n_out=None, activation=None):
        assert n_in is not None or self.n_layer > 0, "n_in must be specified for input layer"
        assert n_out is not None, "n_out must be specified"
        assert activation is not None, "activation must be specified"
 
        if n_in is not None and self.n_layer > 0:
            assert n_in == self.layers[-1].n_out, "n_in does not match with previous layer number of units"
        elif self.n_layer > 0:
            n_in = self.layers[-1].n_out
      
        self.layers += [Layer(n_in, n_out, activation)]
        self.n_layer += 1
 
    def forward(self, X):
        X0 = X
        for layer in self.layers:
            layer.X0 = X0
            layer.forward()
            X0 = layer.X1
        self.Xout = X0
        
    def backward(self, DeltaN):
        Delta1 = DeltaN
        for layer in self.layers[-1::-1]:
            layer.D1 = Delta1
            layer.backward()
            Delta1 = layer.D0

    def updateParam(self, solver_func=None):
        if solver_func is None:
            pass
        else:
            for layer in self.layers[-1::-1]:
                layer.updateParam(solver_func)


class NetTrainer(object):
    """
    NetTrainer iterates over data batches to update Net paramaters that
    minimize a give cost.

    The following are the most important elements necessary to 
    instantiate a NetTrainer object:
    -- net: Net object to be trained
    -- train_data: given in the form a numpy array where the first dimension is the
       number of data exemplars.
    -- label_data: given in a similar form to train_data. Number of exemplars 
       must be consistent with train_data
    -- solver: Solver object that specifies the update rule
    -- loss_func: These function must be chosen from loss_list which is defined in 'loss_functions.py'
    Training parameters are given in the form of a dictionary
    """

    net = None
    batch_size = None
    max_iter = 1000
    solver_func = None
    loss_func = None
    print_interval = None
    train_data = None
    label_data = None

    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

        assert self.net is not None, "Net object cannot be None"
        assert self.loss_func is not None, "No loss was specified"
        self.loss = loss_list[self.loss_func][0]
        self.lossPrime = loss_list[self.loss_func][1]
        assert self.train_data is not None, "Training data must be specified"
        self.data = Data(self.train_data, batch_size=self.batch_size)
        assert self.label_data is not None, "Labels must be specified"        
        self.labels = Data(self.label_data)
        self.solver_func = self.solver.solverFunc()
  
    def train(self, n_iter=None):
        for iTr in range(self.max_iter):
            Xin = self.data.getBatch()
            T = self.labels.getDataAsIn(self.data)
            self.net.forward(Xin)
            objective = np.mean(self.loss(T, self.net.Xout), axis=0)
            self.net.backward(self.lossPrime(T, self.net.Xout) / T.shape[0])
            self.net.updateParam(self.solver_func)
            if iTr % self.print_interval == 0:
                print "Iteration %d, objective = %f" % (iTr,objective)

class Solver(object):
    """
    Solver object contains the method employed to update the network
    parameters based on the gradient information.
    """

    lr_rate = None
    rate_decay = None
    momentum = None
    solver = None
    
    def __init__(self, params):
        for prm_name in params.keys():
            setattr(self, prm_name, params[prm_name])

    def solverFunc(self):
        return getattr(self, self.solver)

    def sgd(self, layer):
        Delta_W = self.momentum * layer.Delta_W - self.lr_rate * np.dot(layer.X0.transpose(), layer.G) 
        Delta_b = self.momentum * layer.Delta_b - self.lr_rate * np.sum(layer.G, axis=0)
        return Delta_W, Delta_b


