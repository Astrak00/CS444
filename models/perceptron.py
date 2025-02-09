"""Perceptron model."""

import numpy as np

DECAY = 0.85

class Perceptron:
    def __init__(self, n_class: int, lr: float, epochs: int, decay: float = 1):
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
        self.reg_const = 8

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
        samples = X_train.shape[0]
        dim = X_train.shape[1]
        
        # Randomize starting weights
        self.w = np.random.rand(dim, self.n_class)
        indices = np.arange(samples)

        for epoch in range(self.epochs):
            # y_pred = X_train @ self.w
            # temp = [np.argmax(i) for i in y_pred]
            # print("Epoch", epoch, "Accuracy",np.sum(y_train == temp) / len(y_train) * 100)

            np.random.shuffle(indices)

            for sample_index in indices:
                for current_class in range(self.n_class):
                    # Only update the weights for incorrect classes
                    if current_class != y_train[sample_index]:
                        if np.dot(np.transpose(self.w)[current_class], X_train[sample_index]) > np.dot(np.transpose(self.w)[y_train[sample_index]], X_train[sample_index]):
                            np.transpose(self.w)[y_train[sample_index]] = np.transpose(self.w)[y_train[sample_index]] + self.lr * X_train[sample_index]
                            np.transpose(self.w)[current_class] = np.transpose(self.w)[current_class] - self.lr * X_train[sample_index]
                    
                    
                    np.transpose(self.w)[current_class] += (self.lr * self.reg_const / samples) * np.transpose(self.w)[current_class]

            # Decrease learning rate
            self.lr *= self.decay

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
