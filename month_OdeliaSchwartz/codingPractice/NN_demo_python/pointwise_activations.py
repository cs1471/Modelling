"""
Wrapper for activation functions and their derivatives
to be used with the DNN classes
"""

import numpy as np


# Logistic
def myLogistic(X):
    return 1.0 / (1.0 + np.exp(-X))

def myLogisticPrime(X):
    S = myLogistic(X)
    return np.multiply(S, 1.0 - S)

# Hyperbolic tangent
def myTanh(X):
    return np.tanh(X)

def myTanhPrime(X):
    return 1 - np.square(np.tanh(X))    

# Rectified linear units
def myRelu(X):
    return np.maximum(0.0, X)

def myReluPrime(X):
    return (X > 0.0).astype(X.dtype)

# Absolute value (fully rectified)
def myAbs(X):
    return np.abs(X)

def myAbsPrime(X):
    return np.sign(X)

# Squared
def mySquare(X):
    return np.square(X)

def mySquarePrime(X):
    return X

# Half squared
def myHalfSquare(X):
    return myRelu(mySquare(X))

def myHalfSquarePrime(X):
    return myRelu(X)

## activation functions are regitered in the following dictionary
func_list = {'sigmoid':(myLogistic, myLogisticPrime),     \
             'tanh':(myTanh, myTanhPrime),        \
             'relu':(myRelu, myReluPrime),        \
             'abs':(myAbs, myAbsPrime),         \
             'square':(mySquare, mySquarePrime),      \
             'half_square':(myHalfSquare, myHalfSquarePrime)}
