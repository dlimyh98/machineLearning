from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    # loop over all training samples (N of them)
    # For a single datapoint, the corresponding row of our gradient matrix is just set appropriately (as derived in the notes)
    # But remember that we are iterating over all datapoints, therefore we utilize increment operations here
    for i in range(num_train):
        scores = X[i].dot(W)                                # Linear function f(x_i,W) = Wx_i; for some image x_i
        correct_class_label = y[i]                          # Extract the training label (which is the correct class label) for the image x_i
        correct_class_score = scores[correct_class_label]   # Extract score that image x_i had for the correct class

        num_didnt_meet_margin = 0

        for j in range(num_classes):
            if j == y[i]:
                continue

            margin = scores[j] - correct_class_score + 1  # note delta = 1

            if margin > 0:
                num_didnt_meet_margin += 1
                loss += margin
                dW[:,j] += X[i]

        dW[:,y[i]] -= (num_didnt_meet_margin * X[i])

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train   # scale gradient ovr the number of samples
    dW += 2 * reg * W # append partial derivative of regularization term

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

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
    num_train = X.shape[0]
    loss = 0.0
    delta = 1
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    ## Semi vectorized implementation from lecture notes
     for i in xrange(num_train):
       scores = X[i].dot(W)
       correct_class_score = scores[y[i]]  # y[i] = c means that X[i] has label c

       # compute the margins for all classes in one vector operation
       margins = np.maximum(0, scores - correct_class_score + delta)

       # on y-th position scores[y] - scores[y] canceled and gave delta. We want
       # to ignore the y-th position and only consider margin on max wrong class
       margins[y[i]] = 0
       loss += np.sum(margins)
    '''

    scores = X.dot(W)                                      # ie. Run linear classifier on minibatch of data
    correct_class_scores = scores[np.arange(num_train), y] # Access scores in a pairwise fashion, using np.arrange(num_train) and y as indices

    # scores = (N,C), correct_class_scores = (N,none)
    # Use np.newaxis to make correct_class_scores = (N,1)
    # Then, broadcasting can happen (1 is stretched to fit C)
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + delta)
    margins[np.arange(num_train),y] = 0    # For some ith image, we need to set margins for it's correct class label to 0. Do this for all N images
    
    loss = np.sum(margins) / num_train  # data loss
    loss += 0.5 * reg * np.sum(W * W)   # regularization

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    '''
    ## Semi-vectorized version. It's not any faster than the loop version.
    num_classes = W.shape[1]
    incorrect_counts = np.sum(margins > 0, axis=1)  # Margins is (N,C). Therefore this gives us an (N,1) matrix reflecting incorrect counts of every image

    for k in xrange(num_classes):
        # use the indices of the margin array as a mask.
        wj = np.sum(X[margins[:, k] > 0], axis=0)
        wy = np.sum(-incorrect_counts[y == k][:, np.newaxis] * X[y == k], axis=0)
        dW[:, k] = wj + wy
    '''

    signal = np.zeros(np.shape(margins))

    signal[margins > 0] = 1    # For every corresponding entry in 'margins' that has non-zero value, we set a TRUE flag in our signal matrix
    incorrect_counts = np.sum(signal, axis=1)           # For every sample, find the total number of classes where margin > 0
    signal[np.arange(num_train),y] = -incorrect_counts  # We previously set all cells in 'margin' that map to the correct class as 0. We repurpose that cell now.
    dW = X.T.dot(signal)

    dW /= num_train # average out weights
    dW += reg*W     # regularize the weights

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW