"""
`Cognit`
========

Cognit is a streamlined neural computation framework and API engineered with Numpy
fine-tuned for advanced machine learning operations and swift adaptability.

importing Cognit via the command:  >>> import cognit

How to use the docstring
----------------------------
Documentation is available one form: the docstrings provided in classes & functions.
Read deepflow classes & deepflow function docstrings for more information

easy setup
----------
TIP - remember: only make one output neuron


- 1. make a instance for deepflow (this will make sense later)

>>> model = cognit.deepflow()

- 2. make a list for defineing your neuron values

>>> layers_ = [x,y,z] # <- x,y,z to represent layers (input,hidden,output)

- 3. make sure you have your array data (numpy recommended)

>>> np.zeros((datax, datay))

- 4. create a layer instance

>>> model_layer = cognit.deepflow.layers() 

- 5. assign neuron values

>>> model_layer.layer(input_size=x,hidden_size=y,output_size=z) # <- x,y,z to represent layer values eg 2,4,1

- 6. train your neurons (with your array data)

>>> model.train_data(
    X=datax,
    y=datay,
    layers=layers_,  # List of layers created using model_layer
    loss_calc="mse",  # or "CE" for cross-entropy
    y_true=None,  # Placeholder (not used in current implementation)
    y_pred=None,  # Placeholder (not used in current implementation)
    learning_rate=0.01, # Placeholder (Default)
    epochs=100, # Placeholder (Default)
) 




"""

import numpy as np




class deepflow:
    
    """
    `deepflow`
    =====

    deepflow is the class used in cognit, its used by Initializing weights and biases with random values
    deepflow also helps with training the neural network with it's `train_data()` function.


    """
    
    def __init__(self) -> None:
        pass    

    class layers:
        """

            `deepflow.layers`
            ----
            contains nessesary functions for creating input, hidden and output layers.
            (NOTE: using layer, it is important you make an instance, eg. `model_package = cognit.deepflow.layers() model_package.layer(input_size=x,hidden_size=y,output_size=z`)
            
            functions:
            
            - `deepflow.layers.layer`
            - `deepflow.layers.activation_layer`
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
            
        
        def denseLayer(self, input_size, output_size, activation="relu") -> None:
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
            
        def flatLayer(self, X):
            """
            `deepflow.layers.flatLayer()`
            ----
            
            Performs the flattening operation.

            Args:
                X (np.ndarray): The input data.

            Returns:
                np.ndarray: The flattened output.
            """
            # Reshape the input data to a single dimension
            return X.flatten()
        
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
        elif activation == "ReLU":
        # Apply ReLU activation
            deepflow.activation.ReLU(X)
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
          
        def ReLU(self, X):
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
                    "relu": self.ReLU,
                    "swish": self.swish,
                    "elu": self.elu,
                    "tanh": self.tanh,
                    "mish": self.mish,
                    "linear": self.linear,
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
                    "relu": self.ReLU,
                    "swish": self.swish,
                    "elu": self.elu,
                    "tanh": self.tanh,
                    "mish": self.mish,
                    "linear": self.linear,
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
            # Clip predicted probabilities to avoid division by zero
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
            # Calculate cross-entropy for each sample
            cross_entropy_loss = -np.sum(y_true * np.log(y_pred), axis=1)
            # Return average cross-entropy
            return np.mean(cross_entropy_loss)
  
    def train_data(self, X, y, layers, loss_calc, y_true, y_pred, learning_rate=0.01, epochs=100):  
        """
        `deepflow.train_data()`
        ----
        Trains the neural network using provided data.

        Args:
            X (numpy.ndarray): The input data.
            y (numpy.ndarray): The target values (labels).
            learning_rate (float, optional): The learning rate used for updating weights and biases. Defaults to 0.01.
            epochs (int, optional): The number of training epochs. Defaults to 100.

        Raises:
            ValueError: If the input data and target values have different shapes.
        """

        # Check if input data and target values have the same shape
        if X != y:
            raise ValueError("Input data and target values must have the same number of samples.")

        for epoch in range(epochs):  # Use the epochs parameter
            # Forward pass
            predictions = deepflow.activation.forward(X)

            # Calculate loss
            if loss_calc == "mse" or loss_calc == "MSE":
                loss = deepflow.losses.mse(y_true, y_pred)
            elif loss_calc == "ce" or loss_calc == "CE" or loss_calc == "cross entropy":
                loss = deepflow.losses.CE(y_true, y_pred)
            else:
                print("Error: loss calculator not specified")
                exit(1)

            # Backward pass to calculate gradients
            gradients = deepflow.activation.backward(X, y, predictions)

            # Update weights and biases based on gradients and learning rate
            # ... (implementation omitted for safety reasons)

            # Print loss for monitoring
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
