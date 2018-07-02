from builtins import range
from builtins import object
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

    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
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
        # order is important here
        W1, W2, b1, b2 = self.params.values()
        l1_scores, l1_cache = affine_relu_forward(X, W1, b1)
        scores, l2_cache = affine_forward(l1_scores, W2, b2)

        # print(scores)

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
        soft_loss, soft_grad = softmax_loss(scores, y)
        loss = soft_loss + (0.5 * self.reg * np.sum(W1 * W1))
        loss += (0.5 * self.reg * np.sum(W2 * W2))

        dl1_scores, dW2, db2 = affine_backward(soft_grad, l2_cache)
        dx, dW1, db1 = affine_relu_backward(dl1_scores, l1_cache)

        dW2 += self.reg * W2
        dW1 += self.reg * W1

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

    def __init__(self, hidden_dims, input_dim=3 * 32 * 32, num_classes=10,
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
        layer_params = [input_dim] + hidden_dims
        layer_params.append(num_classes)
        for index, value in enumerate(layer_params):
            if index == len(layer_params) - 1:
                continue
            j = index + 1
            W_key = "W{}".format(j)
            b_key = "b{}".format(j)

            self.params[W_key] = weight_scale * np.random.randn(value, layer_params[j])
            self.params[b_key] = np.zeros(layer_params[j])

            if self.use_batchnorm and index != len(layer_params) - 2:
                gamma_key = "gamma{}".format(j)
                beta_key = "beta{}".format(j)
                self.params[gamma_key] = np.ones(layer_params[j])
                self.params[beta_key] = np.zeros(layer_params[j])

        # for key, val in self.params.items():
        #     print(key, val.shape)
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
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
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
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

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
        score = None
        caches = {}

        for i in range(self.num_layers):
            j = i + 1
            W_key = "W{}".format(j)
            b_key = "b{}".format(j)
            cache_key = 'cache_{}'.format(j)
            drpout_cache_key = 'drpout_cache_{}'.format(j)

            W, b = self.params[W_key], self.params[b_key]
            if i == self.num_layers - 1:
                scores, cache_last = affine_forward(score, W, b)
                caches[cache_key] = cache_last
            else:
                x_term = X if i == 0 else score
                if self.use_batchnorm:
                    gamma_key = 'gamma{}'.format(j)
                    beta_key = 'beta{}'.format(j)
                    gamma, beta = self.params[gamma_key], self.params[beta_key]
                    score, cache_int = FullyConnectedNet.affine_batchnorm_relu_forward(self, x_term, W, b, gamma, beta, self.bn_params[i])
                else:
                    score, cache_int = affine_relu_forward(x_term, W, b)

                caches[cache_key] = cache_int

                if self.use_dropout:
                    score, cache_drop = dropout_forward(score, self.dropout_param)
                    caches[drpout_cache_key] = cache_drop

        # l1_scores, l1_cache = affine_relu_forward(X, W1, b1)
        # scores, l2_cache = affine_forward(l1_scores, W2, b2)
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
        loss, soft_grad = softmax_loss(scores, y)

        dscores = None

        for i in range(self.num_layers, 0, -1):
            dW = None
            db = None

            W_key = "W{}".format(i)
            b_key = "b{}".format(i)

            cache_key = 'cache_{}'.format(i)
            drpout_cache_key = 'drpout_cache_{}'.format(i)

            W = self.params[W_key]
            loss += 0.5 * self.reg * np.sum(W * W)

            # use affine_backward for last layer
            if i == self.num_layers:
                dscores, dW, db = affine_backward(soft_grad, caches[cache_key])
            else:
                if self.use_dropout:
                    dscores = dropout_backward(dscores, caches[drpout_cache_key])
                if self.use_batchnorm:
                    dscores, dW, db, dgamma, dbeta = FullyConnectedNet.affine_batchnorm_relu_backward(self, dscores, caches[cache_key])
                    gamma_key = 'gamma{}'.format(i)
                    beta_key = 'beta{}'.format(i)
                    grads[gamma_key] = dgamma
                    grads[beta_key] = dbeta
                else:
                    dscores, dW, db = affine_relu_backward(dscores, caches[cache_key])

            dW += self.reg * W
            grads[W_key] = dW
            grads[b_key] = db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def affine_batchnorm_relu_forward(self, x, w, b, gamma, beta, bn_param):
        a, fc_cache = affine_forward(x, w, b)
        bn_out, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
        out, relu_cache = relu_forward(bn_out)
        cache = (fc_cache, bn_cache, relu_cache)
        return out, cache

    def affine_batchnorm_relu_backward(self, dout, cache):
        fc_cache, bn_cache, relu_cache = cache
        da = relu_backward(dout, relu_cache)
        dx_, dgamma, dbeta = batchnorm_backward(da, bn_cache)
        dx, dw, db = affine_backward(dx_, fc_cache)
        return dx, dw, db, dgamma, dbeta
