# -*- coding: utf-8 -*-
"""Exercise 3.

Least Square
"""

import numpy as np
from costs import compute_mse

def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        mse: scalar.

    >>> least_squares(np.array([0.1,0.2]), np.array([[2.3, 3.2], [1., 0.1]]))
    (array([ 0.21212121, -0.12121212]), 8.666684749742561e-33)
    """
    # ***************************************************
    gram = tx.T.dot(tx)
    w = np.linalg.solve(gram, tx.T.dot(y)) # it's good practice to use solve and not inv !!
    mse = compute_mse(y, tx, w)
    
    return w, mse
    # ***************************************************
    raise NotImplementedError
