from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]
    num_classes =  W.shape[1]

    for sample in range(num_samples):
        label = y[sample]
        score_vector = X[sample] @ W

        softmax_vector = np.exp(score_vector - score_vector.max()) / np.sum(np.exp(score_vector - score_vector.max()))
        loss += -np.log(softmax_vector[label])    # Loss is wrt the current sample's class

        # Gradient update
        for clss in range(num_classes):
            dW[:, clss] += (softmax_vector[clss] - (label == clss)) * X[sample]

    # Average the loss and include regularization term
    loss = (loss / num_samples) + (0.5* reg * np.sum(W**2))

    # Average the gradient and include regularization-derivative term
    dW /= num_samples
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_samples = X.shape[0]

    scores = X @ W
    scores_row_maximal = np.max(scores,axis=1,keepdims=True)
    softmax_matrix = np.exp(scores - scores_row_maximal) / np.sum(np.exp(scores - scores_row_maximal), axis=1, keepdims=True)
    subtraction_term = np.zeros_like(softmax_matrix)    # NxC matrix initialized to zeroes

    # Each row of subtraction_term corresponds to some datapoint
    # For each datapoint (row), there is ONE column which should be set to 1 (corresponding to when datapoint's label equals class label)
    subtraction_term[np.arange(num_samples),y] = 1
    Q = softmax_matrix - subtraction_term

    dW = X.T.dot(Q)
    dW /= num_samples    # Average the gradient
    dW += reg * W        # Include regularization-derivative term

    # Each row of softmax_matrix corresponds to softmax_vector for some datapoint n
    # For each datapoint n (ie. each row of softmax_matrix), we take negative log of the datapoint's corresponding label
    datapoint_loss = -np.log(softmax_matrix[np.arange(num_samples), y])    # np.shape(datapoint_loss) == num_samples (which makes sense)
    loss = np.sum(datapoint_loss)    # Sum over all rows of datapoint_loss
    loss = (loss / num_samples) + (0.5* reg * np.sum(W**2))    # Average the loss and include regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW