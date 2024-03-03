# Cognit ![image](./logo.png)

> Cognit is a streamlined neural computation framework and API engineered with Numpy fine-tuned for advanced machine learning operations and swift adaptability.

## easy setup

> TIP - remember: only make one output neuron


1. make a instance for deepflow (this will make sense later)

```
 model = cognit.deepflow()
```

2. make a list for defineing your neuron values

```
layers_ = [x,y,z] # <- x,y,z to represent layers (input,hidden,output)
```

3. make sure you have your array data (numpy recommended)

```
np.zeros((datax, datay))
```

4. create a layer instance

```
model_layer = cognit.deepflow.layers()
```

5. assign neuron values

```
model_layer.layer(input_size=x,hidden_size=y,output_size=z) # <- x,y,z to represent layer values eg 2,4,1
```

6. train your neurons (with your array data)

```
model.train_data(
    X=datax,
    y=datay,
    layers=layers_,  # List of layers created using model_layer
    loss_calc="mse",  # or "CE" for cross-entropy
    y_true=None,  # Placeholder (not used in current implementation)
    y_pred=None,  # Placeholder (not used in current implementation)
    learning_rate=0.01, # Placeholder (Default)
    epochs=100, # Placeholder (Default)
) 
```
