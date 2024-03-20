"""
`Cognit`
========

Cognit is a streamlined neural computation framework and API engineered with Numpy
fine-tuned for advanced machine learning operations and swift adaptability.

importing Cognit via the command:  >>> import cognit as cn

How to use the docstring
----------------------------
Documentation is available one form: the docstrings provided in classes & functions.
Read deepflow classes & deepflow function docstrings for more information

easy setup
----------
TIP - remember: only make one output neuron

- 1. import cognit (for your own dataset, use numpy)
>>> import cognit as cn

- 2. make sure you have your array data, (using mnist)

>>> dataset = cn.deepflow.dataset.mnist

- 3. make it into 2 variables, X_train and y_train
>>> (X_train,y_train), (X_train,y_train) = cn.deepflow.dataset.load()

- 4. make a sequential container

>>> model = cn.deepflow.sequential ([
    cn.deepflow.layers.flatten(X=12),
    cn.deepflow.layers.dense(128,activation="relu"),
    cn.deepflow.layers.dense(10,activation="softmax")
])

- 5. train the Neural network

>>> cn.deepflow.train_data(X=X_train,y=y_train,optimiser="adam",epochs=100)







"""

import numpy as np
import uuid
import time
import os, sys


class deepflow:
    
    """
    `deepflow`
    =====

    deepflow is the class used in cognit, its used by Initializing weights and biases with random values
    deepflow also helps with training the neural network with it's `train_data()` function.

    sub-classes:
    
    - `deepflow.layers()`
    - `deepflow.activation()`
    - `deepflow.optimisers`
    - `deepflow.losses()`

    """
    global layers_
    layers_ = []
    
    def __init__(self) -> None:
        pass

    class layers:
        """

            `deepflow.layers`
            ----
            contains nessesary functions for creating input, hidden and output layers.
            (NOTE: using layer, it is important you make an instance, eg. `model_package = cognit.deepflow.layers() model_package.layer(input_size=x,hidden_size=y,output_size=z`)
            
            functions:
            
            - `deepflow.layers.layer()`
            - `deepflow.layers.activation_layer()`
            - `deepflow.layers.dense()`
            - `deepflow.layers.flatten()`
            - `deepflow.layers.dropout()`
            """
        def __init__(self,input_size=0,hidden_size=0,output_size=0) -> None:
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
        
        activated_output = ""
        
        def layer(self,input_size, hidden_size, output_size) -> None:
            """

            `deepflow.layers.layer()`
            ----
            `input_size: input neurons`
            `hidden_size: hidden neurons`
            `output_size: output neurons`
            """
            
            str(self.input_size)
            str(self.hidden_size)
            str(self.output_size)
            
            # Initialize weights and biases with random values
            self.weights1 = np.random.randn(self.input_size, self.hidden_size)
            self.biases1 = np.zeros((self.hidden_size,))
            self.weights2 = np.random.randn(self.hidden_size, self.output_size)
            self.biases2 = np.zeros((self.output_size,))
            
        
        def dense(self, input_size, output_size, activation="relu") -> None:
            """
            `deepflow.layers.denseLayer()`
            ----
            Initializes the dense layer.

            Args:
                input_size (int): The number of inputs to the layer.
                output_size (int): The number of outputs from the layer.
                activation (str, optional): The activation function to use. Defaults to "relu".
            """
            # Initialize weights and biases with appropriate distribution (e.g., Xavier initialization)
            self.weights = np.random.randn(input_size, output_size) * np.sqrt(2 / (input_size + output_size))
            self.biases = np.zeros(output_size)
            # Store chosen activation function
            self.activation = activation
            
        def flatten(self, X):
            """
            `deepflow.layers.flatten()`
            ----
            
            Performs the flattening operation.

            Args:
                X (np.ndarray): The input data.

            Returns:
                np.ndarray: The flattened output.
            """
            # Reshape the input data to a single dimension
            return X.flatten()
        
        def dropout(X, keep_prob):
            """
            `deepflow.layers.dropout()`
            -----
            Implements a Dropout layer during training.
            Args:
                X: Input data (numpy array).
                keep_prob: Probability of keeping a neuron (1.0 - dropout rate).

            Returns:
                Output data with dropout applied (numpy array).
            """
            # Generate a random mask with values 0 or 1
            mask = np.random.rand(*X.shape) < keep_prob

            # Apply the mask by multiplying with the input
            output = X * mask

            # Invert dropout for training stability (optional)
            # Scaled output to maintain expected value during backpropagation
            output /= keep_prob

            return output
        
        def activation_layer(activation,X):
            
            """
            `deepflow.layers.activation_layer()`
            ----
            Applies a specified activation function to the input data.

        Args:
            activation (str): The name of the activation function to use.
                Supported options include:
                
                    - "sigmoid"
                    - "ReLU"
                    - "tanh"
                    - "elu" (Exponential Linear Unit)
                    - "mish" (Mish activation)
                    - "linear"
                    - "swish" (Swish activation)
            X (numpy.ndarray): The input data.

        Returns:
            numpy.ndarray: The input data after applying the specified activation function.

        Raises:
            ValueError: If an unsupported activation function is provided.
            """
            
            if activation == "sigmoid":
                deepflow.activation.sigmoid(X)
            elif activation == "relu":
            # Apply ReLU activation
                deepflow.activation.relu(X)
            elif activation == "tanh":
                deepflow.activation.tanh(X)
            elif activation == "elu":
                deepflow.activation.elu(X)
            elif activation == "mish":
                deepflow.activation.tanh(X)
            elif activation == "linear":
                deepflow.activation.linear(X)
            elif activation == "swish":
                deepflow.activation.swish(X)
            elif activation == "softmax":
                deepflow.activation.softmax(X)
            else:
                raise ValueError("Unsupported activation function: {}".format(activation))

    class activation:
        """
        `deepflow.activation()`
        ----
        
        contains activation functions nessesary to create layers, or anything else
        functions:
        
        - `deepflow.activation.sigmoid()`
        - `deepflow.activation.ReLU()`
        - `deepflow.activation.elu()`
        - `deepflow.activation.linear()`
        - `deepflow.activation.mish()`
        - `deepflow.activation.tanh()`
        - `deepflow.activation.swish()`
        - `deepflow.activation.forward()`
        - `deepflow.activation.softmax()`
        
        
        """
        def __init__(self) -> None:
            pass
        
        def sigmoid(self, X):
            """
            `deepflow.activation.sigmoid()`
            ----
            `X: input data`

            Applies the sigmoid activation function to the input data.

            Returns:
                The output data after applying the sigmoid function.
            """
            return 1 / (1 + np.exp(-X))
          
        def relu(self, X):
            """
            `deepflow.activation.ReLU()`
            ----
            `X: input data`

            Applies the ReLU activation function to the input data.

            Returns:
                The output data after applying the ReLU function.
            """
            return np.maximum(0, X)  # Maximum of 0 and the input
          
        def elu(self, X, alpha=1.0):
            """
            `deepflow.activation.elu()`
            ----
            `X: input data`
            `alpha: alpha parameter for the ELU function (default: 1.0)`

            Applies the ELU activation function to the input data.

            Returns:
                The output data after applying the ELU function.
            """
            return np.where(X <= 0, alpha * (np.exp(X) - 1), X)
          
        def linear(self, X):
            """
            `deepflow.activation.linear()`
            ----
            `X: input data`

            Applies the linear activation function to the input data (identity function).

            Returns:
                The unmodified input data.
            """
            return X

        def mish(self, X):
            """
            `deepflow.activation.mish()`
            ----
            `X: input data`

            Applies the Mish activation function to the input data.

            Returns:
                The output data after applying the Mish function.
            """
            return X * np.tanh(np.log1p(np.exp(X)))  # Mish formula
          
        def tanh(self, X):
          """
          `deepflow.activation.tanh()`
          ----
          `X: input data`

          Applies the hyperbolic tangent (tanh) activation function to the input data.

          Returns:
              The output data after applying the tanh function.
          """
          return np.tanh(X)

        def softmax(x):
            """
            `deepflow.activation.softmax`
            -----
            
            Computes the softmax of the input vector.

            Args:
                x: Input vector (numpy array).

            Returns:
                Softmax output vector (numpy array).
            """
            # Exponentiate the input values
            exp_x = np.exp(x)

            # Avoid potential overflow by calculating the sum of exponentials first
            sum_exp_x = np.sum(exp_x, axis=0, keepdims=True)

            # Normalize by dividing with the sum to get probabilities
            return exp_x / sum_exp_x
          
        def swish(self, X):
            """
            `deepflow.activation.swish()`
            ----
            `X: input data`

            Applies the Swish activation function to the input data.

            Returns:
                The output data after applying the Swish function.
            """
            return X * self.sigmoid(X)  # X * sigmoid(X)


        def forward(self, X, activation_func="sigmoid"):
          """
          `deepflow.activation.forward()`
          ----
          `X: input data`
          `activation_func: string (optional, defaults to "sigmoid")`

          Performs forward propagation through two layers, applying the specified activation function after the first layer.

          Returns:
              The output of the second layer.
          """
  
          activation_map = {
                    "sigmoid": self.sigmoid,
                    "relu": self.relu,
                    "swish": self.swish,
                    "elu": self.elu,
                    "tanh": self.tanh,
                    "mish": self.mish,
                    "linear": self.linear,
                    "softmax": self.softmax,
                }
          activation_func = activation_map.get(activation_func, self.sigmoid)  # Default to sigmoid if not specified

          layer1 = np.dot(X, self.weights1) + self.biases1
          layer1 = activation_func(layer1)  # Apply chosen activation
          output = np.dot(layer1, self.weights2) + self.biases2
          return output
      
        def backward(self, X, y, output, activation_func="sigmoid"):
            """
            `deepflow.activation.backward()`
            ----
            `X: input data`
            `y: true labels`
            `output: output from the forward propagation`
            `activation_func: string (optional, defaults to "sigmoid")`

            Performs backward propagation using the output of the forward propagation.
            Adjusts the weights and biases based on the error rate.

            Returns:
                The gradients of the weights and biases.
            """

            activation_map = {
                    "sigmoid": self.sigmoid,
                    "relu": self.relu,
                    "swish": self.swish,
                    "elu": self.elu,
                    "tanh": self.tanh,
                    "mish": self.mish,
                    "linear": self.linear,
                    "softmax": self.softmax,
            }
            activation_derivative = activation_map.get(activation_func, self.sigmoid_derivative)

            # Calculate the error
            error = y - output
            d_output = error * activation_derivative(output)

            # Calculate the gradient for weights2 and biases2
            layer1 = np.dot(X, self.weights1) + self.biases1
            layer1 = self.sigmoid(layer1)  # or your chosen activation function
            d_weights2 = np.dot(layer1.T, d_output)
            d_biases2 = np.sum(d_output, axis=0, keepdims=True)

            # Calculate the error for layer1
            d_layer1 = np.dot(d_output, self.weights2.T) * activation_derivative(layer1)

            # Calculate the gradient for weights1 and biases1
            d_weights1 = np.dot(X.T, d_layer1)
            d_biases1 = np.sum(d_layer1, axis=0, keepdims=True)

            # Update the weights and biases
            self.weights1 += d_weights1
            self.biases1 += d_biases1
            self.weights2 += d_weights2
            self.biases2 += d_biases2

            return d_weights1, d_biases1, d_weights2, d_biases2

    class losses:
        """
        `deepflow.losses()`
        ----
        
        contains mse and CE (cross entropy) loss calculators used for training neurons
        functions:
        
        - `deepflow.losses.mse()`
        - `deepflow.losses.CE()` 
        """
        
        def __init__(self) -> None:
            pass
        
        def mse(y_true, y_pred):
            """
            `deepflow.losses.mse()`
            ----
            Calculates the mean squared error between true and predicted values.

            Args:
                y_true (np.ndarray): The true target values.
                y_pred (np.ndarray): The predicted values.

            Returns:
                float: The mean squared error loss.
            """
            return np.mean((y_true - y_pred) ** 2)
        
        def CE(y_true, y_pred, epsilon=1e-10):
            """
            `deepflow.losses.CE()`
            ----
            
            Calculates the cross-entropy loss between true target distribution and predicted probabilities.

            Args:
                y_true (np.ndarray): The one-hot encoded true target distribution.
                y_pred (np.ndarray): The predicted probabilities.
                epsilon (float, optional): A small value to avoid division by zero. Defaults to 1e-10.

            Returns:
                float: The cross-entropy loss.
            """
        def sce(y_true, y_pred):
            """
            Calculates the sparse categorical crossentropy loss.

            Args:
                y_true: Ground truth labels, an integer array of shape (num_samples,).
                y_pred: Predicted probabilities, a float array of shape (num_samples, num_classes).

            Returns:
                The average sparse categorical crossentropy loss across samples.
            """
            # Clip predictions to avoid overflow/underflow in log
            y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

            # One-hot encode the labels
            y_true = np.eye(y_pred.shape[1])[y_true]

            # Calculate the loss per sample
            loss = -np.sum(y_true * np.log(y_pred), axis=1)

            # Return the average loss
            return np.mean(loss)
        
            # Clip predicted probabilities to avoid division by zero
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            # Calculate cross-entropy for each sample
            cross_entropy_loss = -np.sum(y_true * np.log(y_pred), axis=1)
            # Return average cross-entropy
            return np.mean(cross_entropy_loss)

    class optimiser:
        """
        `deepflow.optimisers()`
        ------

        stores built in optimesers.
        functions:
        
        - `deepflow.optimiser.adam()`
        """
        def __init__(self) -> None:
            pass

        def adam(params, grads, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
            """
            `deepflow.optimiser.adam()`
            ------
            
            Implements the Adam optimizer for parameter updates.

            Args:
                params: List of NumPy arrays containing model parameters.
                grads: List of NumPy arrays containing gradients for each parameter.
                learning_rate: Learning rate (default: 0.001).
                beta1: Decay rate for 1st moment estimate (default: 0.9).
                beta2: Decay rate for 2nd moment estimate (default: 0.999).
                epsilon: Small value to prevent division by zero (default: 1e-8).

            Returns:
                List of updated parameters (NumPy arrays).
            """

            # Initialize moment estimates (zeros)
            m_t = [np.zeros_like(p) for p in params]
            v_t = [np.zeros_like(p) for p in params]

            t = 0  # Timestep

            updated_params = []
            for p, g in zip(params, grads):
                t += 1

                # Update moving average of gradient (1st moment)
                m_t[0] = beta1 * m_t[0] + (1 - beta1) * g

                # Update moving average of squared gradient (2nd moment)
                v_t[0] = beta2 * v_t[0] + (1 - beta2) * g**2

                # Bias correction for 1st moment
                m_hat = m_t[0] / (1 - beta1**t)

                # Bias correction for 2nd moment
                v_hat = v_t[0] / (1 - beta2**t)

                # Update parameter with Adam formula
                p -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

                updated_params.append(p)

            return updated_params
    
    class dataset:
        def __init__(self) -> None:
            pass
        
        class mnist:
            def __init__():
                try:
                    os.system("wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
                    print("\nDownloading mnist dataset from tf keras API: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz")
                    
                    
                except:
                    print("\ncognit.deepflow - mnist dataset failed to download.")
        def load(loadi="./mnist.npz"):
            np.load(loadi)
            
    
    def sequential(first_arg, *args, **kwargs):
        """
        `deepflow.sequential()`
        --------
        
        Creates a sequential container(and runs it)

        Args:
            first_arg: The first argument, which should be a list of code blocks or
                        a single code block to be executed.
            *args: Additional positional arguments to be passed to the code blocks.
            **kwargs: Additional keyword arguments to be passed to the code blocks.

        Returns:
            The result of the last executed code block, if any.
        """

        if isinstance(first_arg, list):
            for code_block in first_arg:
                result = eval(code_block, args, kwargs)  # Execute code with args and kwargs
        else:
            result = eval(first_arg, args, kwargs)  # Execute the first argument directly

        return result

            


    def train_data(self, optimiser="adam", X=None, y=None, layers_=[],loss_calc="mse", epochs=100, min_delta=0.001, patience=5):
        """
        `deepflow.train_data()`
        -----
        
        Trains the neural network using provided data with Adam optimizer and early stopping.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values (labels).
            layers (list): List of layer objects defining the network architecture.
            loss_calc (function): Function that calculates the loss between predicted and true values.
            epochs (int, optional): The maximum number of training epochs. Defaults to 100.
            min_delta (float, optional): Minimum change in loss to consider improvement. Defaults to 0.001.
            patience (int, optional): Number of epochs with no improvement to trigger early stopping. Defaults to 5.

        Returns:
            list: List of training losses for each epoch.
        """

        # Check if input data and target values have the same shape
        if X.shape != y.shape:
            raise ValueError("Input data and target values must have the same shape.")

        # Initialize weights and biases for each layer
        for layer in layers_:
            layer.dense()

        # Training loop
        training_losses = []
        best_loss = np.inf
        epochs_no_improvement = 0
        for epoch in range(epochs):
            # Forward propagation
            y_pred = self.forward(X, layers_)  # Use your custom forward function

            # Calculate loss
            loss = loss_calc(y, y_pred)
            training_losses.append(loss)


            #animation = ["10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%"]
            animation = ["[==>                  ] 10%","[====>                ] 20%", "[======>              ] 30%", "[========>            ] 40%", "[==========>          ] 50%", "[============>        ] 60%", "[==============>     ] 70%", "[================>    ] 80%", "[==================>  ] 90%", "[====================>] 100% "]

            for i in range(len(animation)):
                time.sleep(0.2)
                sys.stdout.write("\r" + animation[i % len(animation)])
                sys.stdout.flush()
                print(f"\nEpoch: {epoch+1} - loss: {loss:.4f}")

            # Backpropagation using your custom function
            grads = self.backward(y, y_pred, layers_)  # Use your custom backward function

            # Update weights and biases using Adam optimizer
            if optimiser == "adam":
                updated_params = self.optimiser().adam(
                    [layer.weights for layer in layers_] + [layer.biases for layer in layers_], grads)
            else:
                print("cognit.deepflow.optimisers.adam - sorry, optimiser not supported.")
            
            for i, layer in enumerate(layers_):
                layer.weights = updated_params[2 * i]
                layer.biases = updated_params[2 * i + 1]
