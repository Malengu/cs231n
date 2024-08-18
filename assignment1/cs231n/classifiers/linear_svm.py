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

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        # initializing backprop variables
        dloss = 1/num_train # scalar value
        dscores = np.zeros_like(scores) # (1, C)
        dcorrect_class_score = 0.0

        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            
            if margin > 0:
                loss += margin
                # inner loop backward pass
                dmargin = dloss # scalar value
                dscores[j] += 1.0 * dmargin
                dcorrect_class_score += -1.0 * dmargin # scalar value
                
        # outer loop backward pass
        dscores[y[i]] = np.copy(dcorrect_class_score)
        dW += np.dot(np.expand_dims(X[i], axis=0).T, np.expand_dims(dscores, axis=0)) # (D, C)

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

    dW += 2 * reg * W # (D, C)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, D = X.shape
    scores = np.dot(X, W) # (N, C)
    margins = scores - np.expand_dims(scores[range(N), y], axis=1) + 1 # (N, C)
    margins_clip = np.maximum(0, margins) # (N, C)
    margins_sum = np.sum(margins_clip, axis=1, keepdims=True) - 1 # (N, 1)
    loss = np.mean(margins_sum) + reg*np.sum(W*W) # scalar value

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

    dloss = 1.0
    dmargins_sum = np.ones_like(margins_sum) * (1/N) * dloss # (N, 1)
    dmargins_clip = np.ones_like(margins_clip) * dmargins_sum # (N, C) * (N, 1) -> (N, C)
    dmargins = (margins > 0) * dmargins_clip # (N, C) - backprop through max gate
    dscores = np.ones_like(scores) * dmargins # (N, C)
    dscores[range(N), y] += np.sum(-1.0 * dmargins, axis=1) # (N,)
    dW = np.dot(X.T, dscores) # (D, C)
    dW += 2 * reg * W * dloss # backprop through reg node

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
