"""Neural network model."""

from typing import Sequence

import numpy as np


class NeuralNetwork:
    """A multi-layer fully-connected neural network. The net has an input
    dimension of N, a hidden layer dimension of H, and output dimension C. 
    We train the network with a MLE loss function. The network uses a ReLU
    nonlinearity after each fully connected layer except for the last. 
    The outputs of the last fully-connected layer are passed through
    a sigmoid. 
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        output_size: int,
        num_layers: int,
        opt: str,
    ):
        """Initialize the model. Weights are initialized to small random values
        and biases are initialized to zero. Weights and biases are stored in
        the variable self.params, which is a dictionary with the following
        keys:
        W1: 1st layer weights; has shape (D, H_1)
        b1: 1st layer biases; has shape (H_1,)
        ...
        Wk: kth layer weights; has shape (H_{k-1}, C)
        bk: kth layer biases; has shape (C,)
        Parameters:
            input_size: The dimension D of the input data
            hidden_size: List [H1,..., Hk] with the number of neurons Hi in the
                hidden layer i
            output_size: output dimension C
            num_layers: Number of fully connected layers in the neural network
        """
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.num_layers = num_layers

        self.t = 1
        self.m: dict[str, np.ndarray] = {}
        self.v: dict[str, np.ndarray] = {}

        assert len(hidden_sizes) == (num_layers - 1)
        sizes: list[int] = [input_size] + hidden_sizes + [output_size]

        self.params: dict[str, np.ndarray] = {}
        for i in range(1, num_layers + 1):
            self.params["W" + str(i)] = np.random.randn(sizes[i - 1], sizes[i]) / np.sqrt(sizes[i - 1])
            self.params["b" + str(i)] = np.zeros(sizes[i])

            # TODO: You may set parameters for Adam optimizer here

            self.m["W"+str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.m["b"+str(i)] = np.zeros_like(self.params['b' + str(i)])
            
            self.v["W"+str(i)] = np.zeros_like(self.params['W' + str(i)])
            self.v["b"+str(i)] = np.zeros_like(self.params['b' + str(i)])

    def linear(self, W: np.ndarray, X: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Fully connected (linear) layer.
        Parameters:
            W: the weight matrix
            X: the input data
            b: the bias
        Returns:
            the output
        """
        # result = np.zeros((len(X), len(b)))
        # for i in range(len(X)):
        #     result = np.dot(X[i], W) + b
        # return result
        np.dot(X, W)
        a = X @ W
        return a + b
    
    def linear_grad(self, W: np.ndarray, X: np.ndarray, b: np.ndarray, de_dz: np.ndarray, reg, N) -> np.ndarray:
        """Gradient of linear layer
            z = WX + b
            returns de_dw, de_db, de_dx
        """
        # TODO: implement me
        return 

    def relu(self, X: np.ndarray) -> np.ndarray:
        """Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output
        """
        return np.maximum(X, 0)

    def relu_grad(self, X: np.ndarray) -> np.ndarray:
        """Gradient of Rectified Linear Unit (ReLU).
        Parameters:
            X: the input data
        Returns:
            the output data
        """
        return np.where(X > 0, 1, 0)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x: np.ndarray) -> np.ndarray:
        return x * (1 - x)

    def mse(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.mean((y - p) ** 2)
    
    def mse_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return -2*(y-p) / y.shape[0]
    
    def mse_sigmoid_grad(self, y: np.ndarray, p: np.ndarray) -> np.ndarray:
        return (p - y) * self.sigmoid_grad(p)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute the outputs for all of the data samples.
        Hint: this function is also used for prediction.
        Parameters:
            X: Input data of shape (N, D). Each X[i] is a training or
                testing sample
        Returns:
            Matrix of shape (N, C) 
        """
        self.outputs = {}
        # TODO: implement me. You'll want to store the output of each layer in
        # self.outputs as it will be used during back-propagation. You can use
        # the same keys as self.params. You can use functions like
        # self.linear, self.relu, and self.mse in here.

        self.outputs["r"+str(0)] = X
        current_X = X

        for layer_index in range(1, self.num_layers + 1):
            current_X = self.linear(self.params["W" + str(layer_index)], current_X, self.params["b" + str(layer_index)])
            self.outputs["l" + str(layer_index)] = current_X

            # If it is not the last layer, apply ReLU
            if layer_index != (self.num_layers):
                current_X = self.relu(current_X)
                self.outputs["r" + str(layer_index)] = current_X
            
            # If it is the last layer, apply sigmoid
            else:
                current_X = self.sigmoid(current_X)
                self.outputs["s" + str(layer_index)] = current_X

        return current_X

    def backward(self, y: np.ndarray) -> float:
        """Perform back-propagation and compute the gradients and losses.
        Parameters:
            y: training value targets
        Returns:
            Total loss for this batch of training samples
        """
        self.gradients = {}
        # TODO: implement me. You'll want to store the gradient of each
        # parameter in self.gradients as it will be used when updating each
        # parameter and during numerical gradient checks. You can use the same
        # keys as self.params. You can add functions like self.linear_grad,
        # self.relu_grad, and self.softmax_grad if it helps organize your code.
        
        final_prediction = self.outputs["s" + str(self.num_layers)]

        # Calculate the loss
        loss = self.mse(y, final_prediction)

        mse_grad = self.mse_grad(y, final_prediction)

        grad = self.sigmoid_grad(final_prediction) * mse_grad

        self.gradients["s" + str(self.num_layers)] = grad

        for layer_index in range(self.num_layers, 0, -1):
            weight = self.params["W" + str(layer_index)]
            bias = self.params["b" + str(layer_index)]

            # Calculate the gradient of the weights
            
            input_X = self.outputs["r" + str(layer_index - 1)]
            grad_W = input_X.T @ grad
            grad_b = np.ones(grad.shape[0]).T @ grad
            grad_X = grad @ weight.T

            self.gradients["W" + str(layer_index)] = grad_W / self.output_size
            self.gradients["b" + str(layer_index)] = grad_b / self.output_size

            # Calculate the gradient of the ReLU instead of the first one, which is a sigmoid
            if layer_index > 1:
                input_X = self.outputs["l" + str(layer_index - 1)]
                grad_Z = self.relu_grad(input_X) 
                grad = grad_X * grad_Z

        return loss

    def update(
        self,
        lr: float = 0.001,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
        opt: str = "SGD",
    ):
        """Update the parameters of the model using the previously calculated
        gradients.
        Parameters:
            lr: Learning rate
            b1: beta 1 parameter (for Adam)
            b2: beta 2 parameter (for Adam)
            eps: epsilon to prevent division by zero (for Adam)
            opt: optimizer, either 'SGD' or 'Adam'
        """
        # TODO: implement me. You'll want to add an if-statement that can
        # handle updates for both SGD and Adam depending on the value of opt.
        if opt == "SGD":
            for i in range(1, self.num_layers + 1):
                self.params["W" + str(i)] -= lr * self.gradients["W" + str(i)]
                self.params["b" + str(i)] -= lr * self.gradients["b" + str(i)]
        elif opt == "Adam":
            for i in range(1, self.num_layers+1):
                
                self.m["W"+str(i)] = b1*self.m["W"+str(i)] + (1-b1)*self.gradients["W" + str(i)]
                self.m["b"+str(i)] = b1*self.m["b"+str(i)] + (1-b1)*self.gradients["b" + str(i)]
                
                self.v["W"+str(i)] = b2*self.v["W"+str(i)] + (1-b2)*np.square(self.gradients["W" + str(i)])
                self.v["b"+str(i)] = b2*self.v["b"+str(i)] + (1-b2)*np.square(self.gradients["b" + str(i)])
                m_W_hat = self.m["W" + str(i)] / (1 - b1 ** self.t)
                v_W_hat = self.v["W" + str(i)] / (1 - b2 ** self.t)
                self.params["W" + str(i)] -= lr*np.divide(m_W_hat, (np.sqrt(v_W_hat) + eps) )
                m_b_hat = self.m["b" + str(i)] / (1 - b1 ** self.t)
                v_b_hat = self.v["b" + str(i)] / (1 - b2 ** self.t)
                delta_b = lr*np.divide(m_b_hat, (np.sqrt(v_b_hat) + eps) )
                delta_b = delta_b.reshape(-1)
                self.params["b" + str(i)] -= delta_b
                self.t +=1
        
        return