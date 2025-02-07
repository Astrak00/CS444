"""Support Vector Machine (SVM) model."""

import numpy as np


class SVM:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
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

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculate gradient of the svm hinge loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            the gradient with respect to weights w; an array of the same shape
                as w
        """
        N, _ = X_train.shape
        grad_w = np.zeros_like(self.w)

        for i in range(N):
            scores = X_train[i].dot(self.w)
            correct_class_score = scores[y_train[i]]
            for j in range(self.n_class):
                if j == y_train[i]:
                    continue
                margin = scores[j] - correct_class_score + 1  # delta = 1
                if margin > 0:
                    grad_w[:, j] += X_train[i]
                    grad_w[:, y_train[i]] -= X_train[i]

        grad_w /= N
        grad_w += self.reg_const * self.w  # Regularization term
        return grad_w

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
        _, D = X_train.shape
        self.w = np.random.rand(D, self.n_class)

        for _ in range(self.epochs):
            grad_w = self.calc_gradient(X_train, y_train)
            self.w -= self.lr * grad_w

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
        scores = X_test.dot(self.w)
        return np.argmax(scores, axis=1)
