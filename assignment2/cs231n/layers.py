from builtins import range
import numpy as np
from numpy.lib import triu_indices_from
from numpy.ma.extras import mask_rowcols


def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, *D = x.shape
    x_rows = np.reshape(x, (N, np.prod(D))) # (N, D)
    out = np.dot(x_rows, w) + b # (N, M)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, *D = x.shape
    x = np.reshape(x, (N, np.prod(D))) # (N, D)
    dx = np.reshape(np.dot(dout, w.T), (N, *D)) # (N, D)
    dw = np.dot(x.T, dout) # (D, M)
    db = np.sum(dout, axis=0, keepdims=True) # (1, M)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    out =  np.maximum(0, x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dx = (x > 0) * dout 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # compute softmax loss
    eps = 1e-17
    N, C = x.shape
    
    x_norm = x - np.max(x, axis=1, keepdims=True) # (N, C)
    counts = np.exp(x_norm) # (N, C)
    counts_sum = np.sum(counts, axis=1, keepdims=True) # (N, 1)
    probs = counts * counts_sum**-1 # (N, C)
    lossi = -np.log(probs[range(N), y] + eps) # (N, 1)
    loss = np.mean(lossi) # data loss

    # backward pass
    dloss = 1.0
    dlossi = np.ones_like(lossi) * (1/N) * dloss # (N, 1)
    dprobs = np.zeros_like(probs) # (N, C)
    dprobs[range(N), y] = (-1.0 / (probs[range(N), y] + eps)) * dlossi # (N, 1)
    dcounts_sum = np.sum(-counts * counts_sum**-2 * dprobs, axis=1, keepdims=True) # (N, 1)
    dcounts = counts_sum**-1 * dprobs # (N, C)
    dcounts += np.ones_like(counts) * dcounts_sum # (N, C)
    dx_norm = counts * dcounts # (N, C)
    dx = np.ones_like(x) * dx_norm # (N, C)
    dx[range(N), np.argmax(x, axis=1)] += np.sum(-dx_norm, axis=1) # (N, C)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        meani = np.mean(x, axis=0, keepdims=True) # (1, D)
        vari = np.mean((x - meani)**2, axis=0, keepdims=True) # (1, D)
        xi = (x - meani) / (np.sqrt(vari + eps)) # (N, D)
        out = gamma * xi + beta # (N, D)

        # keep track of running mean and variance
        running_mean = momentum * running_mean + (1 - momentum) * meani
        running_var = momentum * running_var + (1 - momentum) * vari

        # caching intermediates for backward pass
        cache = (gamma, xi, x, meani, vari, eps, N)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        xi = (x - running_mean) / (np.sqrt(running_var + eps))
        out = gamma * xi + beta

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpacking the cache
    gamma, xi, x, meani, vari, eps, N = cache 

    # backward pass through batchnorm params' nodes
    dgamma = np.sum(xi * dout, axis=0, keepdims=True) # (1, D)
    dbeta = np.sum(dout, axis=0, keepdims=True) # (1, D)
    # backward pass through batchnorm intermediates
    dxi = gamma * dout # (N, D)
    dvari = np.sum(0.5 * (meani - x) * (vari + eps)**(-3/2) * dxi, axis=0, keepdims=True) # (1, D)
    dmeani = np.sum(-1.0 * (np.sqrt(vari + eps))**-1 * dxi, axis=0, keepdims=True) # (1, D)
    dmeani += np.sum((2 / N) * (meani - x) * dvari, axis=0, keepdims=True) # (1, D)
    dx = (np.sqrt(vari + eps))**-1 * dxi # (N, D)
    dx += (2/ N) * (x - meani) * dvari # (N, D)
    dx += (1 / N) * np.ones_like(x) * dmeani # (N, D)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # unpacking the cache
    gamma, xi, x, meani, vari, eps, N = cache
    
    dgamma = np.sum(xi * dout, axis=0, keepdims=True) # (1, D)
    dbeta = np.sum(dout, axis=0, keepdims=True) # (1, D)
    dx = (1.0/(N*np.sqrt(vari + eps))) * (-xi*np.sum(xi*dout*gamma, axis=0) - np.sum(dout*gamma, axis=0) + N*dout*gamma) # (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
   
    meani = np.mean(x, axis=1, keepdims=1) # (N, 1)
    vari = np.mean((x - meani)**2, axis=1, keepdims=True) # (N, 1)
    xi = (x - meani) / np.sqrt(vari + eps) # (N, D)
    out = gamma * xi + beta # (N, D)

    # caching intertmediates for backprop
    cache = gamma, out, xi, x, meani, vari, eps

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # meani = np.mean(x, axis=1, keepdims=1) # (D, 1)
    # vari = np.mean((x - meani)**2, axis=1, keepdims=True) # (D, 1)
    # xi = (x - meani) / np.sqrt(vari + eps) # (N, D)
    # out = gamma * xi + beta # (N, D)

    # # caching intertmediates for backprop
    # cache = gamma, out, xi, x, meani, vari, eps
    

    gamma, out, xi, x, _, vari, eps = cache
    N, D = x.shape
    stdi = np.sqrt(vari + eps)

    dgamma = np.sum(xi*dout, axis=0, keepdims=True) # (1, D)
    dbeta = np.sum(dout, axis=0, keepdims=True) # (1, D)
    dxi = dout * gamma # (N, D)
    dx = (1.0/(D*stdi)) * (-xi*np.sum(xi*dxi, axis=1, keepdims=True) - np.sum(dxi, axis=1, keepdims=True) + D*dxi) # (N, D)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        mask =  (np.random.rand(*x.shape) < p) / p # dropout mask
        out = x * mask # drop!

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        out = x

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        dx = mask * dout

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modify the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # conv layer hyperparameters!
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape

    # spatial sizes of the output volume
    H_out = 1 + (H + 2*pad - HH) // stride
    W_out = 1 + (W + 2*pad - WW) // stride
    # padding input!
    if pad > 0:
      pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
      x_mat = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    else:
      x_mat = np.copy(x)

    # initialize out with all zeros
    out = np.zeros((N, F, H_out, W_out), dtype=x.dtype)
    # convolutional layer forward pass!
    for f in range(F):
      for i in range(H_out):
        for j in range(W_out):
          istart, jstart = i*stride, j*stride
          iend, jend = istart + HH, jstart + WW
          out[:, f, i, j] = np.sum(x_mat[:, :, istart:iend, jstart:jend] * w[f], axis=(1, 2, 3)) + b[f]

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, w, b, conv_param = cache
    # convolutional layer hyperparameters!
    stride = conv_param.get("stride", 1)
    pad = conv_param.get("pad", 0)

    F, C, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape

    # padding input!
    if pad > 0:
      pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
      x_mat = np.pad(x, pad_width=pad_width, mode='constant', constant_values=0)
    else:
      x_mat = np.copy(x)

    # initializing grads with zeros!
    dx_mat, dw, db = np.zeros_like(x_mat), np.zeros_like(w), np.zeros_like(b)

    # convolutional layer backward pass!
    for f in range(F):
      for i in range(H_out):
        for j in range(W_out):
          istart, jstart = i*stride, j*stride
          iend, jend = istart + HH, jstart + WW
          dx_mat[:, :, istart:iend, jstart:jend] += w[f] * np.expand_dims(dout[:, f, i, j], axis=(1, 2, 3))
          dw[f] += np.sum(x_mat[:, :, istart:iend, jstart:jend] * np.expand_dims(dout[:, f, i, j], axis=(1, 2, 3)), axis=0)
          db[f] += np.sum(dout[:, f, i, j])
    
    # unpadding the grad for dx!
    dx = dx_mat if pad == 0 else dx_mat[:, :, pad:-pad, pad:-pad]
          
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # pooling layer hyperparameters!
    pool_height = pool_param.get("pool_height", 2)
    pool_width = pool_param.get("pool_width", 2)
    stride = pool_param.get("stride", 2)

    N, C, H, W = x.shape

    # spatial sizes of the output volume
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # initializing output volume with zeros
    out = np.zeros((N, C, H_out, W_out), dtype=x.dtype)
    # pooling layer operation for forward pass!
    for i in range(H_out):
      for j in range(W_out):
        istart, jstart = i*stride, j*stride
        iend, jend = istart + pool_height, jstart + pool_width
        out[:, :, i, j] = np.max(x[:, :, istart:iend, jstart:jend], axis=(2, 3))

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    x, pool_param = cache

    # pooling layer hyperparameters!
    pool_height = pool_param.get("pool_height", 2)
    pool_width = pool_param.get("pool_width", 2)
    stride = pool_param.get("stride", 2)

    N, C, H_out, W_out = dout.shape

    # initializing dx with zeros
    dx = np.zeros_like(x)
    # pooling layer operation for backward pass!
    for i in range(H_out):
      for j in range(W_out):
        istart, jstart = i*stride, j*stride
        iend, jend = istart + pool_height, jstart + pool_width

        window = x[:, :, istart:iend, jstart:jend]

        # masking out grads!
        mask = np.zeros_like(window)

        for bidx, b in enumerate(np.split(window, N, axis=0)):
          for cidx, c in enumerate(np.split(b, C, axis=1)):
            mask[bidx, cidx, np.argmax(c)//pool_height, np.argmax(c)%pool_width] = 1

        dx[:, :, istart:iend, jstart:jend] += np.expand_dims(dout[:, :, i, j], axis=(2, 3)) * mask


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    x = np.swapaxes(x, 1, -1).reshape(-1, C) # swap axes
    out, cache = batchnorm_forward(x, gamma, beta, bn_param)
    out = out.reshape(N, W, H, C).swapaxes(1, -1) # swap back axes

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # batchnorm_backward_alt(dout, cache):
    N, C, H, W = dout.shape
    dout = np.swapaxes(dout, 1, -1).reshape(-1, C) # match vanilla backprop
    dx, dgamma, dbeta = batchnorm_backward_alt(dout, cache)
    dx = dx.reshape(N, W, H, C).swapaxes(1, -1)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N, C, H, W = x.shape
    assert C%G == 0, "improper no. of groups per datapoint!"

    x = np.reshape(x, (N*G, H*W*(C//G))) # reshape each datapoint into groups
    # reshaping gamma and beta to match the vanilla layernorm API
    gamma = np.tile(np.tile(gamma, H*W).reshape(G, H*W*(C//G)), (N, 1)) 
    beta = np.tile(np.tile(beta, H*W).reshape(G, H*W*(C//G)), (N, 1))

    out, cache = layernorm_forward(x, gamma, beta, gn_param)
    cache = (G, *cache) # update cache
    out = np.reshape(out, (N, C, H, W)) # reshaping back into original output shape
  
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # gamma, out, xi, x, _, vari, eps = cache
    # N, D = x.shape
    # stdi = np.sqrt(vari + eps)

    # dgamma = np.sum(xi*dout, axis=0, keepdims=True) # (1, D)
    # dbeta = np.sum(dout, axis=0, keepdims=True) # (1, D)

    N, C, H, W = dout.shape
    G = cache[0]

    assert C%G == 0,"improper no. of groups!"

    dout = np.reshape(dout, (N*G, H*W*(C//G)))

    gamma, out, xi, x, _, vari, eps = cache[1:]
    M, D = x.shape

    # backprop through x node!
    stdi = np.sqrt(vari + eps)
    dxi = dout * gamma # (N, D)
    dx = (1.0/(D*stdi)) * (-xi*np.sum(xi*dxi, axis=1, keepdims=True) - np.sum(dxi, axis=1, keepdims=True) + D*dxi) # (N, D)
    # reshape grads into correct shapes!
    dx = np.reshape(dx, (N, C, H, W))

    # backprop through beta and gamma node!
    dgamma_i = np.reshape(xi*dout, (-1)) # (M, D)
    dbeta_i = np.reshape(dout, (-1)) # (M, D)
    # initializing gamma and beta
    dgamma, dbeta = np.zeros((1, C, 1, 1)), np.zeros((1, C, 1, 1))
    ws = 0
    C_prime = C//G

    # looping through and summing up contributions to each filter weights!
    for g in range(N*G):
      for i in range((g%G)*C_prime, (g%G)*C_prime+C_prime):
        dgamma[0, i] += np.sum(dgamma_i[ws*H*W: ws*H*W+(H*W)])
        dbeta[0, i] += np.sum(dbeta_i[ws*H*W: ws*H*W+(H*W)])
        ws += 1
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
