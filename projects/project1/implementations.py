import numpy as np

###################################################################################################################
# Loss calculation functions used throughout the file
###################################################################################################################


def compute_loss_mse(y, tx, w):
    """Compute the loss using MSE.

    Args:
        y: shape=(N, ).
        tx: shape=(N,D).
        w: shape=(D,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    loss = 1 / (2 * tx.shape[0]) * np.sum((y - tx.dot(w)) ** 2)
    return loss


def compute_loss_LR(y, tx, w):
    """Compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)

    Returns:
        a non-negative loss
    """
    loss_scaled = np.squeeze(-y.T.dot(tx.dot(w))) + np.sum(
        np.log(1 + np.exp(tx.dot(w)))
    )
    return 1 / y.shape[0] * loss_scaled


###################################################################################################################
# Mean squared error gradient descent
###################################################################################################################


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, ).
        tx: shape=(N,D).
        initial_w: shape=(D, ). Initialization for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.

    Returns:
        w: final w vector, shape (D,) where D is the number of features.
        loss: loss value for the final w vector.
    """

    w = initial_w
    for _ in range(max_iters):
        grad = -1 / y.shape[0] * tx.T.dot(y - tx.dot(w))
        w = w - gamma * grad

    loss = compute_loss_mse(y, tx, w)
    return w, loss


###################################################################################################################
# Mean squared error stochastic gradient descent
###################################################################################################################


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):

    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = (
            np.random.randint(max_batches, size=num_batches) * batch_size
        )  # max_batches is the minimum value and num_batches gives the number of random data we want
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: final w vector, shape (D,) where D is the number of features.
        loss: loss value (scalar) for the final w vector.
    """

    w = initial_w

    for _ in range(max_iters):
        for y_b, tx_b in batch_iter(y, tx, batch_size=1, num_batches=1):
            grad = -1 / y_b.shape[0] * tx_b.T.dot(y_b - tx_b.dot(w))
            w = w - gamma * grad

    loss = compute_loss_mse(y, tx, w)

    return w, loss


###################################################################################################################
# Least squares
###################################################################################################################


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
    w: optimal weights, shape (D,) where D is the number of features.
    loss: loss value (scalar) for optimal w vector.
    """

    gram = tx.T.dot(tx)
    w = np.linalg.solve(gram, tx.T.dot(y))
    loss = compute_loss_mse(y, tx, w)

    return w, loss


###################################################################################################################
# Ridge regression
###################################################################################################################


def ridge_regression(y, tx, lambda_):
    """Calculates the ridge regression solution.

    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        lambda_: scalar.

    Returns:
        w: optimal weights, shape (D,) where D is the number of features.
        loss: loss value (scalar) for optimal w vector.
    """

    lambda_prime = 2 * y.shape[0] * lambda_
    left = tx.T.dot(tx) + lambda_prime * np.identity(tx.shape[1])
    right = tx.T.dot(y)
    w = np.linalg.solve(left, right)
    loss = compute_loss_mse(y, tx, w)

    return w, loss


###################################################################################################################
# Logistic regression
###################################################################################################################


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """
    Logistic regression using gradient descent algorithm.

    Args:
        y: shape=(N, ).
        tx: shape=(N,D) where D is the number of features.
        initial_w: shape=(D, ). Initialization for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.

    Returns:
        w: final w vector, shape (D,).
        loss: loss value for the final w vector.
    """
    w = initial_w
    for _ in range(max_iters):
        pred = tx.dot(w)
        sigmoid_pred = np.exp(pred) / (1 + np.exp(pred))
        grad = 1 / y.shape[0] * tx.T.dot(sigmoid_pred - y)
        w -= gamma * grad
    loss = compute_loss_LR(y, tx, w)

    return w, loss


###################################################################################################################
# Regularized logistic regression
###################################################################################################################


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """
    Regularized logistic regression using gradient descent algorithm.

    Args:
        y: shape=(N, ).
        tx: shape=(N,D) where D is the number of features.
        lambda_: a scalar denoting the regularization parameter.
        initial_w: shape=(D, ). Initialization for the model parameters.
        max_iters: a scalar denoting the total number of iterations of GD.
        gamma: a scalar denoting the stepsize.

    Returns:
        w: final w vector, shape (D,).
        loss: loss value for the final w vector.
    """
    w = initial_w
    for _ in range(max_iters):
        pred = tx.dot(w)
        sigmoid_pred = np.exp(pred) / (1 + np.exp(pred))
        grad = 1 / y.shape[0] * tx.T.dot(sigmoid_pred - y) + 2 * lambda_ * w
        w -= gamma * grad
    loss = compute_loss_LR(y, tx, w)

    return w, loss
