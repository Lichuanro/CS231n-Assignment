import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin        
        dW[:, j] += X[i, :].T
        dW[:, y[i]] -= X[i, :].T
        

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradient.
  dW += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros_like(W) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  score = np.dot(X, W)
  
  y_mat = np.zeros_like(score)

  for y_i in y:
    y_mat[y_i, range(score.shape[1])] = 1
        
  correct_score = np.multiply(y_mat, score)
  sums = np.sum(correct_score, axis=0)
        
  margins = score - sums + 1
        
  result = np.maximum(0, margins)
  result = np.sum(result, axis=1) - 1
        
  loss = np.sum(result) / float(score.shape[0])
  loss += reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
        
  # make each entry 1 if it is > 0, 0 otherwise
  margins[margins > 0] = 1
  margins[margins < 0] = 0

  # keep margins mostly the same but for each column, zero out the row corresponding to the
  # correct label
  # (basically change the 1's to 0's, since we are doing w_y*x - w_y*x + 1 for those entries)
  for y_i in y:
    margins[y_i, range(score.shape[1])] = 0

  # compute column sums and then set the elements that we zeroed out above to the negative of
  # that sum
  col_sums = np.sum(margins, axis=0)

  for y_i in y:
    margins[y_i, range(score.shape[1])] = -1.0 * col_sums

  dW = np.dot(X.T, margins)

  dW /= float(score.shape[1])

  # Add regularization to the gradient
  dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
