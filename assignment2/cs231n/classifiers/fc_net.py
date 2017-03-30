import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network with ReLU nonlinearity and
  softmax loss that uses a modular layer design. We assume an input dimension
  of D, a hidden dimension of H, and perform classification over C classes.
  
  The architecure should be affine - relu - affine - softmax.

  Note that this class does not implement gradient descent; instead, it
  will interact with a separate Solver object that is responsible for running
  optimization.

  The learnable parameters of the model are stored in the dictionary
  self.params that maps parameter names to numpy arrays.
  """
  
  def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
               weight_scale=1e-3, reg=0.0):
    """
    Initialize a new network.

    Inputs:
    - input_dim: An integer giving the size of the input
    - hidden_dim: An integer giving the size of the hidden layer
    - num_classes: An integer giving the number of classes to classify
    - dropout: Scalar between 0 and 1 giving dropout strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - reg: Scalar giving L2 regularization strength.
    """
    self.params = {}
    self.reg = reg
    
    ############################################################################
    # TODO: Initialize the weights and biases of the two-layer net. Weights    #
    # should be initialized from a Gaussian with standard deviation equal to   #
    # weight_scale, and biases should be initialized to zero. All weights and  #
    # biases should be stored in the dictionary self.params, with first layer  #
    # weights and biases using the keys 'W1' and 'b1' and second layer weights #
    # and biases using the keys 'W2' and 'b2'.                                 #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
    self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b1'] = np.zeros(hidden_dim)
    self.params['b2'] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################


  def loss(self, X, y=None):
    """
    Compute loss and gradient for a minibatch of data.

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
    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the two-layer net, computing the    #
    # class scores for X and storing them in the scores variable.              #
    ############################################################################
    H, cache_lay1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
    scores, cache_lay2 = affine_forward(H, self.params['W2'], self.params['b2'])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If y is None then we are in test mode so just return scores
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the two-layer net. Store the loss  #
    # in the loss variable and gradients in the grads dictionary. Compute data #
    # loss using softmax, and make sure that grads[k] holds the gradients for  #
    # self.params[k]. Don't forget to add L2 regularization!                   #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    W1 , b1 = self.params['W1'], self.params['b1']
    W2 , b2 = self.params['W2'], self.params['b2']
    loss, dscores = softmax_loss(scores, y)
    loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2))
    dH, dW2, db2 = affine_backward(dscores, cache_lay2)
    dX, dW1, db1 = affine_relu_backward(dH, cache_lay1)
    dW1 += self.reg * W1
    dW2 += self.reg * W2  
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


class FullyConnectedNet(object):
  """
  A fully-connected neural network with an arbitrary number of hidden layers,
  ReLU nonlinearities, and a softmax loss function. This will also implement
  dropout and batch normalization as options. For a network with L layers,
  the architecture will be
  
  {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax
  
  where batch normalization and dropout are optional, and the {...} block is
  repeated L - 1 times.
  
  Similar to the TwoLayerNet above, learnable parameters are stored in the
  self.params dictionary and will be learned using the Solver class.
  """

  def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
               dropout=0, use_batchnorm=False, reg=0.0,
               weight_scale=1e-2, dtype=np.float32, seed=None):
    """
    Initialize a new FullyConnectedNet.
    
    Inputs:
    - hidden_dims: A list of integers giving the size of each hidden layer.
    - input_dim: An integer giving the size of the input.
    - num_classes: An integer giving the number of classes to classify.
    - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
      the network should not use dropout at all.
    - use_batchnorm: Whether or not the network should use batch normalization.
    - reg: Scalar giving L2 regularization strength.
    - weight_scale: Scalar giving the standard deviation for random
      initialization of the weights.
    - dtype: A numpy datatype object; all computations will be performed using
      this datatype. float32 is faster but less accurate, so you should use
      float64 for numeric gradient checking.
    - seed: If not None, then pass this random seed to the dropout layers. This
      will make the dropout layers deteriminstic so we can gradient check the
      model.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_layers = 1 + len(hidden_dims)
    self.dtype = dtype
    self.params = {}

    ############################################################################
    # TODO: Initialize the parameters of the network, storing all values in    #
    # the self.params dictionary. Store weights and biases for the first layer #
    # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
    # initialized from a normal distribution with standard deviation equal to  #
    # weight_scale and biases should be initialized to zero.                   #
    #                                                                          #
    # When using batch normalization, store scale and shift parameters for the #
    # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
    # beta2, etc. Scale parameters should be initialized to one and shift      #
    # parameters should be initialized to zero.                                #
    ############################################################################
    self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
    self.params['b1'] = np.zeros(hidden_dims[0])
    if self.use_batchnorm:
        self.params['gamma1'] = np.ones(input_dim)
        self.params['beta1'] = np.zeros(input_dim)
    for i in range(1, self.num_layers-1):
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(hidden_dims[i-1], hidden_dims[i])
        self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
        if self.use_batchnorm:
            self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i-1])
            self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i-1])
        
    self.params['W'+str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[self.num_layers-2], num_classes)
    self.params['b'+str(self.num_layers)] = np.zeros(num_classes)

    #print self.params
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # When using dropout we need to pass a dropout_param dictionary to each
    # dropout layer so that the layer knows the dropout probability and the mode
    # (train / test). You can pass the same dropout_param to each dropout layer.
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    # With batch normalization we need to keep track of running means and
    # variances, so we need to pass a special bn_param object to each batch
    # normalization layer. You should pass self.bn_params[0] to the forward pass
    # of the first batch normalization layer, self.bn_params[1] to the forward
    # pass of the second batch normalization layer, etc.
    self.bn_params = []
    if self.use_batchnorm:
      self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]
    
    # Cast all parameters to the correct datatype
    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
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
    #print X.shape
    X = X.reshape(X.shape[0],-1)
    #print X.shape
    if self.use_batchnorm:
        H_bn = [None] * (self.num_layers-1)
        H_aff = [None] * self.num_layers
        H_relu = [None] * (self.num_layers-1)
        cache_aff = [None] * self.num_layers
        cache_relu = [None] * self.num_layers
        cache_bn = [None] * (self.num_layers-1)
        if self.use_dropout:
            H_drop = [None] * (self.num_layers-1)
            cache_drop = [None] * self.num_layers

        H_bn[0],  cache_bn[0] = batchnorm_forward(X, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
        H_aff[0], cache_aff[0] = affine_forward(H_bn[0], self.params['W1'], self.params['b1'])
        H_relu[0], cache_relu[0] = relu_forward(H_aff[0])
        if self.use_dropout:
            H_drop[0], cache_drop[0] = dropout_forward(H_relu[0],self.dropout_param)            

        for i in range(1, self.num_layers-1):
            if self.use_dropout:
                H_bn[i],  cache_bn[i] = batchnorm_forward(H_drop[i-1], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                H_aff[i], cache_aff[i] = affine_forward(H_bn[i], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                H_relu[i], cache_relu[i] = relu_forward(H_aff[i])
                H_drop[i], cache_drop[i] = dropout_forward(H_relu[i],self.dropout_param) 
            else:
                H_bn[i],  cache_bn[i] = batchnorm_forward(H_relu[i-1], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
                H_aff[i], cache_aff[i] = affine_forward(H_bn[i], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                H_relu[i], cache_relu[i] = relu_forward(H_aff[i])
            
                

        H_aff[self.num_layers-1], cache_aff[self.num_layers-1] = affine_forward(H_relu[self.num_layers-2],self.params['W' + str(self.num_layers)],self.params['b' + str(self.num_layers)])
        scores = H_aff[self.num_layers-1]
    else:
        H = [None] * self.num_layers
        cache = [None] * self.num_layers
        if self.use_dropout:
            H_drop = [None] * (self.num_layers-1)
            cache_drop = [None] * self.num_layers
            
        H[0],  cache[0] = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        if self.use_dropout:
            H_drop[0], cache_drop[0] = dropout_forward(H[0],self.dropout_param) 
        for i in range(1, self.num_layers-1):
            if self.use_dropout:
                H[i], cache[i] = affine_relu_forward(H_drop[i-1], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
                H_drop[i], cache_drop[i] = dropout_forward(H[i], self.dropout_param) 
            else:
                H[i], cache[i] = affine_relu_forward(H[i-1], self.params['W'+str(i+1)], self.params['b'+str(i+1)])

        if self.use_dropout:
            H[self.num_layers-1], cache[self.num_layers-1] = affine_forward(H_drop[self.num_layers-2],self.params['W' + str(self.num_layers)],self.params['b' + str(self.num_layers)])
            scores = H[self.num_layers-1]
        else:
            H[self.num_layers-1], cache[self.num_layers-1] = affine_forward(H[self.num_layers-2],self.params['W' + str(self.num_layers)],self.params['b' + str(self.num_layers)])
            scores = H[self.num_layers-1]
        

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    if self.use_batchnorm:
        dX = [None] * (self.num_layers+1)
        loss, dX[self.num_layers] = softmax_loss(scores, y)

        dX[self.num_layers-1], grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dX[self.num_layers], cache_aff[self.num_layers-1])
        grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]
        loss += 0.5 * self.reg * np.sum(self.params['W'+str(self.num_layers)]*self.params['W'+str(self.num_layers)])

        for i in range(self.num_layers-2,-1,-1):
            if self.use_dropout:
                dX_drop = dropout_backward(dX[i+1], cache_drop[i])
                dX_relu = relu_backward(dX_drop, cache_relu[i])
                dX_aff, grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_backward(dX_relu, cache_aff[i])
                dX[i], grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = batchnorm_backward(dX_aff, cache_bn[i])
                loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
                grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
            else:
                dX_relu = relu_backward(dX[i+1], cache_relu[i])
                dX_aff, grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_backward(dX_relu, cache_aff[i])
                dX[i], grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = batchnorm_backward(dX_aff, cache_bn[i])
                loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
                grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
    else:  
        dX = [None] * (self.num_layers+1)
        loss, dX[self.num_layers] = softmax_loss(scores, y)

        dX[self.num_layers-1], grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dX[self.num_layers], cache[self.num_layers-1])
        grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]
        loss += 0.5 * self.reg * np.sum(self.params['W'+str(self.num_layers)]*self.params['W'+str(self.num_layers)])
        for i in range(self.num_layers-2,-1,-1):
            if self.use_dropout:
                dX_drop = dropout_backward(dX[i+1], cache_drop[i])
                dX[i], grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_relu_backward(dX_drop, cache[i])
                loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
                grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
            else:
                dX[i], grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_relu_backward(dX[i+1], cache[i])
                loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
                grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads

  def loss_only_with_batch_norm(self, X, y=None):
    """
    Compute loss and gradient for the fully-connected net.

    Input / output: Same as TwoLayerNet above.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.dropout_param is not None:
      self.dropout_param['mode'] = mode   
    if self.use_batchnorm:
      for bn_param in self.bn_params:
        bn_param[mode] = mode

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the fully-connected net, computing  #
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
    H_bn = [None] * (self.num_layers-1)
    H_aff = [None] * self.num_layers
    H_relu = [None] * (self.num_layers-1)
    cache_aff = [None] * self.num_layers
    cache_relu = [None] * self.num_layers
    cache_bn = [None] * (self.num_layers-1)
    
    H_bn[0],  cache_bn[0] = batchnorm_forward(X, self.params['gamma1'], self.params['beta1'], self.bn_params[0])
    H_aff[0], cache_aff[0] = affine_forward(H_bn[0], self.params['W1'], self.params['b1'])
    H_relu[0], cache_relu[0] = relu_forward(H_aff[0])
    
    for i in range(1, self.num_layers-1):
        
        H_bn[i],  cache_bn[i] = batchnorm_forward(H_relu[i-1], self.params['gamma'+str(i+1)], self.params['beta'+str(i+1)], self.bn_params[i])
        H_aff[i], cache_aff[i] = affine_forward(H_bn[i], self.params['W'+str(i+1)], self.params['b'+str(i+1)])
        H_relu[i], cache_relu[i] = relu_forward(H_aff[i])
        
    H_aff[self.num_layers-1], cache_aff[self.num_layers-1] = affine_forward(H_relu[self.num_layers-2],self.params['W' + str(self.num_layers)],self.params['b' + str(self.num_layers)])
    scores = H_aff[self.num_layers-1]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    # If test mode return early
    if mode == 'test':
      return scores

    loss, grads = 0.0, {}
    ############################################################################
    # TODO: Implement the backward pass for the fully-connected net. Store the #
    # loss in the loss variable and gradients in the grads dictionary. Compute #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    #                                                                          #
    # When using batch normalization, you don't need to regularize the scale   #
    # and shift parameters.                                                    #
    #                                                                          #
    # NOTE: To ensure that your implementation matches ours and you pass the   #
    # automated tests, make sure that your L2 regularization includes a factor #
    # of 0.5 to simplify the expression for the gradient.                      #
    ############################################################################
    dX = [None] * (self.num_layers+1)
    loss, dX[self.num_layers] = softmax_loss(scores, y)
    
    dX[self.num_layers-1], grads['W'+str(self.num_layers)], grads['b'+str(self.num_layers)] = affine_backward(dX[self.num_layers], cache_aff[self.num_layers-1])
    grads['W'+str(self.num_layers)] += self.reg * self.params['W'+str(self.num_layers)]
    loss += 0.5 * self.reg * np.sum(self.params['W'+str(self.num_layers)]*self.params['W'+str(self.num_layers)])
    
    for i in range(self.num_layers-2,-1,-1):
        dX_temp = relu_backward(dX[i+1], cache_relu[i])
        dX_temp_2, grads['W'+str(i+1)], grads['b'+str(i+1)] = affine_backward(dX_temp, cache_aff[i])
        dX[i], grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = batchnorm_backward(dX_temp_2, cache_bn[i])
        
        loss += 0.5 * self.reg * np.sum(self.params['W'+str(i+1)]*self.params['W'+str(i+1)])
        grads['W'+str(i+1)] += self.reg * self.params['W'+str(i+1)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads