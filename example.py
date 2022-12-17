# this is an example for using the neural network model 

from neural_network import Model 


# creating a model if activations are None by defualt softmax will be used

# the shape
shape = (2, 4, 4, 4, 2)
# the activations 
activations = ("sigmoid", "ReLU", "softmax", "tanh")
# create the actual model
model = Model(shape, 0.01, activations)
# get the output of the model 
inputs = [0.1, 0.2] # the length is the same as the first number in the shape
output = model.predict(inputs)
print(output)

