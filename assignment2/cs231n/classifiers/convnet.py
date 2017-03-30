import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MultiLayerConvNet(object):
    """
    This class implements a multi-layer convolutional network. It is writen to be
    a general convnet class allowing the user to decide about the architecture
    through the API. The general architecture in this class has the form:
    
    {{conv - [space batch norm] - relu} x N - pool} x M - 
    {affine - [batch norm] - relu} x L - affine - [softmax or SVM]
    
    The network operates in minibatches of data that have shape (N, C, H, W)
    consiting of N images, each with height H and width W and with C input channels.
    
    Currently, this class implements the following 5-layer architecture:
    {conv - space batch norm - relu} x 3 - pool 
    - affine - batchnorm - relu - affine - softmax
    """
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                dtype=np.float32, use_spacebatchnorm=False, use_batchnorm=False, 
                loss_fun='softmax', num_conv_layers=3, num_pool_layers=1,
                num_aff_layers=2):
        """
        Initialize the new network.
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer
        - weight_scale: Scalar giving standard deviation for random initialization of 
          weights
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation
        - use_spacebatchnorm: Whether or not the network should use space batch 
          normalization for convolutional layers
        - use_batchnorm: Whether or not the network should use batch normalization 
          for fully connected layers
        - loss_fun: Specify the loss function used in the network (available SVM 
          or softmax)
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize the architecture of the general convnet
        self.use_spacebatchnorm = use_spacebatchnorm
        self.use_batchnorm = use_batchnorm
        self.num_aff_layers = num_aff_layers 
        self.num_conv_layers = num_conv_layers 
        self.num_pool_layers = num_pool_layers
        self.loss_fun = loss_fun
        
        ############################################################################
        #################### Network parameters initialization #####################
        
        # Initialize wieghts and biases
        self.params['W1'] = weight_scale*np.random.randn(num_filters,input_dim[0],
                                                         filter_size,filter_size)
        self.params['W2'] = weight_scale*np.random.randn(num_filters,num_filters,
                                                         filter_size,filter_size)
        self.params['W3'] = weight_scale*np.random.randn(num_filters,num_filters,
                                                         filter_size,filter_size)
        self.params['W4'] = weight_scale * np.random.randn(input_dim[1] * input_dim[2]
                                                          * num_filters / 4, hidden_dim)
        self.params['W5'] = weight_scale*np.random.randn(hidden_dim, num_classes)
        
        self.params['b1'] = np.zeros(num_filters)
        self.params['b2'] = np.zeros(num_filters)
        self.params['b3'] = np.zeros(num_filters)
        self.params['b4'] = np.zeros(hidden_dim)
        self.params['b5'] = np.zeros(num_classes)
            
        # Initialize space batchnorm parameters
        if self.use_spacebatchnorm:
            self.params['gamma1'] = np.ones(num_filters)
            self.params['gamma2'] = np.ones(num_filters)
            self.params['gamma3'] = np.ones(num_filters)
            self.params['beta1'] = np.zeros(num_filters)
            self.params['beta2'] = np.zeros(num_filters)
            self.params['beta3'] = np.zeros(num_filters)
            
        # Initialize batchnorm parameters
        if self.use_batchnorm:
            self.params['gamma4'] = np.ones(hidden_dim)
            self.params['beta4'] = np.zeros(hidden_dim)
        
        ############################################################################
        
        # Initialize bn_param object used in batch normalization
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_aff_layers-1)]
            
        # Initialize sbn_param object used in space batch normalization
        self.sbn_params = []
        if self.use_spacebatchnorm:
            self.sbn_params = [{'mode': 'train'} for i in xrange(self.num_conv_layers
                                                                 * self.num_pool_layers)]
            
        # Cast all parameters to the correct datatype
        for k, v in self.params.iteritems():
            self.params[k] = v.astype(dtype)
    
    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the general convnet.
        
        Input / output follows the same API as ThreeLayerConvNet in cnn.py 
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        
        # Set train/test mode for batchnorm params and spacebatchnorm params
        # since they behave differently during training and testing.
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param[mode] = mode
        if self.use_spacebatchnorm:
            for sbn_param in self.sbn_params:
                sbn_param[mode] = mode
        
        # pass conv_param to the forward pass for the convolutional layer
        filter_size = (self.params['W1']).shape[2]
        conv_param = {'stride': 1, 'pad':  (filter_size - 1) / 2}
        
        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        
        # Unpack weigths and baises
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']
        W4, b4 = self.params['W4'], self.params['b4']
        W5, b5 = self.params['W5'], self.params['b5']
        
        # Unpack parameters for spacebatchnorm
        gamma1, beta1 = self.params['gamma1'], self.params['beta1']
        gamma2, beta2 = self.params['gamma2'], self.params['beta2']
        gamma3, beta3 = self.params['gamma3'], self.params['beta3']
        gamma4, beta4 = self.params['gamma4'], self.params['beta4']
        
        scores = None
        
        #############################################################################
        ################### Forward pass for the general convnet ####################
        
        # Current implementation defines a 5 layer network
        Conv_lay1, cache_lay1 = conv_spacebatchnorm_relu_forward(X, W1, b1, gamma1, beta1,
                                                                 conv_param, sbn_param)
        Conv_lay2, cache_lay2 = conv_spacebatchnorm_relu_forward(Conv_lay1, W2, b2, 
                                                    gamma2, beta2, conv_param, sbn_param)
        
        Conv_lay3, cache_lay3 = conv_spacebatchnorm_relu_pool_forward(Conv_lay2, W3, b3, 
                                        gamma3, beta3, conv_param, pool_param, sbn_param)
        Aff_lay, cache_lay4 = affine_batchnorm_relu_forward(Conv_lay3, W4, b4, gamma4,
                                                            beta4, bn_param)
        scores, cache_lay5 = affine_forward(Aff_lay, W5, b5)
        
        #############################################################################
        
        if y is None:
            return scores
        
        loss, grads = 0, {}
        
        #############################################################################
        ################### Backward pass for the general convnet 
        if self.loss_fun == 'softmax':
            loss, dscores = softmax_loss(scores, y)
        elif self.loss_fun == 'SVM':
            loss, dscores = svm_loss(scores, y)
        loss += 0.5 * self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3) 
                                  + np.sum(W4*W4) + np.sum(W5*W5))
        
        dAff_lay, dW5, db5 = affine_backward(dscores, cache_lay5)
        dConv_lay3, dW4, db4, dgamma4, dbeta4 = affine_batchnorm_relu_backward(dAff_lay,
                                                                               cache_lay4)
        dConv_lay2, dW3, db3, dgamma3, dbeta3 = conv_spacebatchnorm_relu_pool_backward(
                                                dConv_lay3, cache_lay3)
        dConv_lay1, dW2, db2, dgamma2, dbeta2 = conv_spacebatchnorm_relu_backward(
                                                dConv_lay2, cache_lay2)
        dX, dW1, db1, dgamma1, dbeta1 = conv_spacebatchnorm_relu_backward(dConv_lay1,
                                                                          cache_lay1)
        
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        dW3 += self.reg * W3
        dW4 += self.reg * W4
        dW5 += self.reg * W5
        
        grads['W1'] = dW1
        grads['W2'] = dW2
        grads['W3'] = dW3
        grads['W4'] = dW4
        grads['W5'] = dW5
        grads['b1'] = db1
        grads['b2'] = db2
        grads['b3'] = db3
        grads['b4'] = db4
        grads['b5'] = db5
        grads['gamma1'] = dgamma1
        grads['gamma2'] = dgamma2
        grads['gamma3'] = dgamma3
        grads['gamma4'] = dgamma4
        grads['beta1'] = dbeta1
        grads['beta2'] = dbeta2
        grads['beta3'] = dbeta3
        grads['beta4'] = dbeta4
        
        #############################################################################
        
        return loss, grads
    
    
pass