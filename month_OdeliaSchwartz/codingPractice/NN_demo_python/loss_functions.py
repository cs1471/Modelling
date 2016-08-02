"""
Some commonly employed losses for neural net training
"""

import numpy as np

# Cross- entropy for binomial distributions
def myXent(T, Y):
    return -(np.multiply(T, np.log(Y)) + np.multiply(1 - T, np.log(1 - Y)))

def myXentPrime(T, Y):
    return (Y - T)

# Mean Squared Error
def myMeanSquaredError(T, Y):
    return np.sum(np.square(T - Y), axis=1) / 2.0

def myMeanSquaredErrorPrime(T, Y):
    return (Y - T)

# Hinge loss (only for binary classification problems with {-1, +1} labels)
def myHinge(T, Y):
    return np.maximum(0,  1.0 - np.multiply(T, Y))

def myHingePrime(T, Y):
    return np.multiply((np.multiply(Y, T) < 1.0).astype(Y), -T)

## Losses are registered in the following dictionary
loss_list = {'xent':(myXent, myXentPrime),     \
             'mse':(myMeanSquaredError, myMeanSquaredErrorPrime), \
             'hinge': (myHinge, myHingePrime)}
