# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np


def compute_loss(y, tx, w):
    loss = 1/(2*tx.shape[0]) * np.sum((y - tx.dot(w))**2) # only one dimension for the sum => no need to specify axis along which we want to sum
    return loss

def least_squares(y, tx):
    """calculate the least squares."""
    # ***************************************************
    gram = tx.T.dot(tx)
    w = np.linalg.solve(gram, tx.T.dot(y)) # it's good practice to use solve and not inv !!
    mse = compute_loss(y, tx, w)
    
    return w, mse
    # returns mse, and optimal weights
    # ***************************************************
    raise NotImplementedError
