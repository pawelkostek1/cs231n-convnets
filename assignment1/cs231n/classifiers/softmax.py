import numpy as np
from random import shuffle

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
  
  ## Naive implementation of the loss
  scores = X.dot(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  L = np.zeros(num_train)
  f = np.zeros(num_classes)
  """
  def fn(z,j):
    return np.exp(z[j])/sum(np.exp(z))

  for i in range(num_train):
        for j in range(num_classes):
            f[j] = fn(scores[j,:],j)
        log_C = np.max(f)
        f -= log_C
        L[i] = -fn(scores[i,:],y[i])-log_C+np.log(np.sum(np.exp(f)))
  """
  for i in xrange(num_train):
        f = scores[i,:]
        log_C = -np.max(f)
        f += log_C
        L[i] = np.log(np.sum(np.exp(f)))-f[y[i]]
  
  
  # Sum over all example's losses
  loss = np.sum(L)
  # Normalize the loss function
  loss /= float(num_train)
  # Add regularization term
  loss += 0.5*reg*np.sum(W*W)
  
  ## Naive implementation of the gradient
  #print X.shape, (X[1,:]).shape, W.shape, scores.shape
  for i in xrange(num_train):
        f = scores[i,:]
        log_C = -np.max(f)
        f += log_C
        for j in xrange(num_classes):
            if  j == y[i]:
                dW[:,j] -= X[i]
            dW[:,j] += X[i]*np.exp(f[j])/(np.sum(np.exp(f)))
                
  # Normalize the gradient
  dW /= float(num_train)
  # Add regularization term
  dW += reg*W
   
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  
  ## Vectorized implementation of loss function
  scores = X.dot(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]
  f = scores
  f -= (np.max(scores,axis=1)).reshape(-1,1)
         
  loss = np.sum(np.log(np.sum(np.exp(f),axis=1)))-np.sum(f[xrange(num_train),y])
  
  # Normalize the loss function
  loss /= float(num_train)
  # Add regularization term
  loss += 0.5*reg*np.sum(W*W)
    
  ##Vectorized implementation of gradient
  P = np.zeros((num_train, num_classes), np.int)
  P[np.arange(num_train), y] = 1
  dW -= (X.T).dot(P)
  
  f = np.exp(f)
  f /= (np.sum(f,axis=1)).reshape(-1,1)
  dW += (X.T).dot(f)
  # Normalize the gradient
  dW /= float(num_train)
  # Add regularization term
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

