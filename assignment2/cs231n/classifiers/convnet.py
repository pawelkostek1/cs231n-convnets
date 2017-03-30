import numpy as np

from cs231n.layers import *
frin cs231n.fast_layers import *
from cs231n.layer_utils import *


class MultiLayerConvNet(object):
    """
    This class implements a multi-layer convolutional network. It is writen to be
    a general convnet class allowing the user to decide about the architecture
    through the API. The general architecture in this class has the form:
    
    {{conv - [space batch norm] - relu} x N - pool} x M - 
    {affine - [batch norm]} x L - [softmax or SVM]
    
    The network operates in minibatches of data that have shape (N, C, H, W)
    consiting of N images, each with height H and width W and with C input channels.
    """
    
    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=3,
                hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                dtype=np.float32, , use_spacebatchnorm=False):
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
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        # Initialize the architecture of the general convnet
        self.use_spacebatchnorm = use_spacebatchnorm
        self.use_batchnorm = use_batchnorm
        self.num_afflayers = 0 
        self.num_convlayers = 0 
        slef.num_poollayers = 0 
        
        ############################################################################
        #################### Weights and biases initialization #####################
        
        
        ############################################################################
        
        # Initialize bn_param object used in batch normalization
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in xrange(self.num_afflayers)]
            
        # Initialize sbn_param object used in space batch normalization
        self.sbn_params = []
        if self.use_spacebatchnorm:
            self.sbn_params = [{'mode'}: 'train' for i in xrange(self.num_convlayers * self.poollayers)]
            
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
        
        scores = None
        
        #############################################################################
        ################### Forward pass for the general convnet #####################
        
        #############################################################################
        
        if y is None:
            return scores
        
        loss, grads = 0, {}
        
        #############################################################################
        ################### Backward pass for the general convnet ###################
        
        #############################################################################
        
        return loss, grads
    
    
pass