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

    N, D = X.shape
    D, C = W.shape

    for i in range(N):
      logits = np.dot(X[i], W) # (1, D) @ (D, C) --> (1, C)
      
      counts = np.zeros_like(logits) # (1, C)
      sum_counts = 0.0
 
      for j in range(C):
        counts[j] = np.exp(logits[j]) # (1, C)
        sum_counts += counts[j] # scalar value

      probs = counts / sum_counts # (1, C)
      loss += -np.log(probs[y[i]])

      # outer loop backprob
      dloss = 1 / N
      dprobs = np.zeros_like(probs) # (1, C)
      dprobs[y[i]] = (-1.0 / probs[y[i]]) * dloss
      dcounts = (sum_counts**-1) * dprobs # (1, C)
      dsum_counts = np.sum(-counts * sum_counts**-2 * dprobs) # scalar value
      # inner loop backprop
      dcounts += np.ones_like(counts) * dsum_counts 
      dlogits = counts * dcounts # (1, C)
      # outer loop backprop
      dW += np.dot(np.expand_dims(X[i], axis=0).T, np.expand_dims(dlogits, axis=0)) # (D, C)

    loss /= N
    loss += reg * np.sum(W**2)

    dW += 2 * reg * W # backprop through the regularization node

      
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

    # forward pass
    N, D = X.shape
    logits = np.dot(X, W) # (N, D) @ (D, C) --> (N, C)
    logits = logits - np.max(logits, axis=1, keepdims=True) # (N, C)
    counts = np.exp(logits) # (N, C)
    sum_counts = np.sum(counts, axis=1, keepdims=True) # (N, 1)
    probs = counts / sum_counts # (N, C)
    nll = -np.log(probs[range(N), y]) # (N, 1)
    data_loss = np.mean(nll)
    loss = data_loss + reg * np.sum(W**2)

    # backward pass
    dloss = 1.0
    ddata_loss = np.copy(dloss)
    dnll = np.ones_like(nll) * (1 / N) * ddata_loss # (N, 1)
    dprobs = np.zeros_like(probs) # (N, C)
    dprobs[range(N), y] = (-1 / probs[range(N), y]) * dnll # (N, 1)
    dcounts = sum_counts**-1 * dprobs # (N, C)
    dsum_counts = np.sum(-counts * sum_counts**-2 * dprobs, axis=1, keepdims=True) # (N, 1)
    dcounts += (np.ones_like(counts) * dsum_counts) # (N, C)
    dlogits = counts * dcounts # (N, C)
    dlogits[range(N), np.argmax(logits, axis=1)] += np.sum(-dlogits, axis=1)
    dW = np.dot(X.T, dlogits) # (D, C)
    dW += 2 * reg * W * dloss # backprop through reg node
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
