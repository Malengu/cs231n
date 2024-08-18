from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # weights initializations
        for l in range(self.num_layers):
          
          if l == self.num_layers - 1:
            # parameters for output layer
            self.params['W%d'%(l+1)] = np.random.randn(input_dim, num_classes) * weight_scale
            self.params['b%d'%(l+1)] = np.zeros((1, num_classes))
          else:
            # parameters for hidden layers
            self.params['W%d'%(l+1)] = np.random.randn(input_dim, hidden_dims[l]) * weight_scale
            self.params['b%d'%(l+1)] = np.zeros((1, hidden_dims[l]))
            # parameters for batch/layer norm 
            if self.normalization in ["batchnorm", "layernorm"]:
              self.params['gamma%d'%(l+1)] = np.ones((1, hidden_dims[l])) 
              self.params['beta%d'%(l+1)] = np.zeros((1, hidden_dims[l]))
            # set next layer input size
            input_dim = hidden_dims[l]
  
        
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        caches = {} # book-keeping the caches

        for l in range(self.num_layers):
          w, b = self.params['W%d'%(l+1)], self.params['b%d'%(l+1)]
          # forward pass through output affine layer
          if l == self.num_layers-1:
             scores,  cache = affine_forward(X, w, b)
          else:
            # forward pass through hidden affine-relu layer
            if self.normalization == None:
              out, cache = affine_relu_forward(X, w, b)
            # forward pass through hidden affine-bn/ln-relu layer
            if self.normalization in ["batchnorm", "layernorm"]:
              gamma, beta = self.params['gamma%d'%(l+1)], self.params['beta%d'%(l+1)]
              out, cache = affine_gn_relu_forward(X, w, b, gamma, beta, self.bn_params[l], self.normalization)
            if self.use_dropout:
              out, dropout_cache = dropout_forward(out, self.dropout_param)
              caches['d%d'%(l+1)] = dropout_cache
          # update the input 
          X = out
          # book-keeping cache for each layer
          caches['l%d'%(l+1)] = cache

       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        data_loss, dout = softmax_loss(scores, y) # data loss
        reg_loss = 0.5 * self.reg * np.sum([np.sum(self.params['W%d'%(l+1)]**2) for l in range(self.num_layers)]) # regularization loss
        loss = data_loss + reg_loss

        # backward pass through multi-layer fully connected neural network!
        dloss = 1.0
        ddata_loss = 1.0 * dloss
        dreg_loss = 1.0 * dloss

        # backprop through the layers!
        for l in reversed(range(self.num_layers)):
          # retrieving cache for each layer!
          cache = caches['l%d'%(l+1)]
         
          if l == self.num_layers-1:
            # backprop through final affine layer!
            dout, dw, db = affine_backward(dout, cache)
          else:
            if self.use_dropout:
              # backprop through dropout layer!
              dout = dropout_backward(dout, caches['d%d'%(l+1)])
            if self.normalization == None:
              # backprop through hidden affine-relu layer!
              dout, dw, db = affine_relu_backward(dout, cache)
            if self.normalization in ["batchnorm", "layernorm"]:
              # backprop through hidden affine-bn/ln-relu layer!
              dout, dw, db, dgamma, dbeta = affine_gn_relu_backward(dout, cache, self.normalization)
              grads['gamma%d'%(l+1)], grads['beta%d'%(l+1)] = dgamma, dbeta
          # update params
          grads['W%d'%(l+1)], grads['b%d'%(l+1)] = dw, db 
            
        # backprop through the regularization nodes
        for l in range(self.num_layers):
          grads['W%d'%(l+1)] += self.reg * self.params['W%d'%(l+1)] * dreg_loss

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
