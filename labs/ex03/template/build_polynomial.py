# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.

    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.

    Returns:
        poly: numpy array of shape (N,d+1)"""
    
    # ***************************************************
    poly = np.ones((len(x), degree + 1))
    for exp in range(1, degree+1):
        poly[:, exp] = np.power(x, exp)
    return poly
    # ***************************************************
    raise NotImplementedError
