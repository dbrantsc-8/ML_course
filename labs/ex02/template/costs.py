# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # MSE
    loss = 1/(2*tx.shape[0]) * np.sum((y - tx.dot(w))**2) # only one dimension for the sum => no need to specify axis along which we want to sum
    return loss

    # MAE
    #loss = 1/(tx.shape[0]) * np.sum(np.abs(y - tx.dot(w)))
    #return loss
    # ***************************************************
    raise NotImplementedError
