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
    lossq = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train = X.shape[0]
    num_classes = W.shape[1]
    dim = W.shape[0]

    for i in xrange(num_train):
        scores = X[i].dot(W)
        scores -= np.max(scores)
        correct_class_score = scores[y[i]]
        loss += (-correct_class_score + np.log(np.sum(np.exp(scores))))

        # Compute gradient
        # Here we are computing the contribution to the inner sum for a given i.
        p = np.exp(scores)/np.sum(np.exp(scores))
        # for k in range(num_classes):
        #     dW[:, k] += (p[k] - (k == y[i])) * X[i]
        new_X = X[i][:, np.newaxis]
        grad = p * new_X
        correct_class_grad = ((p[y[i]] - 1) * new_X)
        grad[:, y[i]] = np.concatenate(correct_class_grad)
        dW += grad

    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    dW /= num_train
    dW += reg * W

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
    num_train = X.shape[0]
    scores = X.dot(W)
    sc_iter = range(0, num_train)

    scores -= np.max(scores, axis=1)[:, np.newaxis]
    cc_scores = scores[sc_iter, y]

    diff = - np.log(np.exp(cc_scores)/np.sum(np.exp(scores), axis=1))
    loss += np.sum(diff)
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    p = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    p_sc = p[sc_iter, y] - 1

    # set the correct class scores to zero as they will be elsewhere
    p[sc_iter, y] = 0
    dW += X.T.dot(p)

    int_zer = np.zeros(p.shape)
    int_zer[sc_iter, y] = p_sc

    # add the gradient for the correct class scores to dW
    dW += X.T.dot(int_zer)
    dW /= num_train
    dW += reg * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
