"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float, decrease_lr: float = 1, batch_size: int = 100):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None 
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.decrease_lr = decrease_lr
        self.batch_size = batch_size

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        N = X_train.shape[0]
        fs = X_train.dot(self.w)

        feature_scores = np.exp(fs - np.max(fs, axis=1, keepdims=True))
        probs = feature_scores / np.sum(feature_scores, axis=1, keepdims=True)

        prob_scores = probs
        prob_scores[range(N), y_train] -= 1
        prob_scores /= N

        dW = X_train.T.dot(prob_scores)
        dW += self.reg_const * self.w

        return dW

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        Hint: operate on mini-batches of data for SGD.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.
        
        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        samples = X_train.shape[0]
        dim = X_train.shape[1]

        # Set random weight
        self.w = np.random.rand(dim, self.n_class)

        # mini-batches for SGD stochastic gradient descent
        batch_size = self.batch_size
        batches = samples // batch_size
        indices = np.arange(samples)

        for _ in range(self.epochs):
            # Gradient descent
            np.random.shuffle(indices) # shuffle indices for each epoch 
            for batch in range(batches):
                start = batch * batch_size 
                end = (batch + 1) * batch_size
                if end >= samples:
                    end = -1
                batch_indices = indices[start:end]
                batch_X,batch_y = X_train[batch_indices], y_train[batch_indices]
                batch_w = self.calc_gradient(batch_X, batch_y)
                self.w -= self.lr * batch_w

            # Decrease learning rate
            self.lr *= self.decrease_lr


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Use the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        y_pred = X_test @ self.w
        return [np.argmax(i) for i in y_pred]
