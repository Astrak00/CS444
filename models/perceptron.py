"""Perceptron model."""

import numpy as np

DECAY = 0.95

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, decay: float = False):
        """Initialize a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
        """
        self.w = None
        self.lr = lr
        self.epochs = epochs
        self.n_class = n_class
        self.decay = decay

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the classifier.

        - Use the perceptron update rule as introduced in the Lecture.
        - Initialize self.w as a matrix with random values sampled uniformly from [-1, 1)
        and scaled by 0.01. This scaling prevents overly large initial weights,
        which can adversely affect training.

        Parameters:
            X_train: a number array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        N, D = X_train.shape
        self.w = np.random.rand(self.n_class, D)

        for _ in range(self.epochs):
            if self.decay:
                self.lr *= DECAY
            for i in range(N):
                xi, yi = X_train[i], y_train[i]
                scores = np.dot(self.w, xi)
                predicted_class = np.argmax(scores)

                if predicted_class != yi:
                    self.w[yi] += self.lr * xi
                    self.w[predicted_class] -= self.lr * xi

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
        scores = np.dot(X_test, self.w.T)
        predicted_labels = np.argmax(scores, axis=1)
        return predicted_labels
