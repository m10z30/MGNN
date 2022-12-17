# this is an example for using the neural network model 

from neural_network import Model 
import numpy as np


# creating a model if activations are None by defualt softmax will be used

# the shape
shape = (2, 4, 4, 4, 2)
# the activations 
activations = ("sigmoid", "sigmoid", "sigmoid", "tanh")
# create the actual model
model = Model(shape, 0.01, activations)
# get the output of the model 
inputs = [0.1, 0.2] # the length is the same as the first number in the shape
output = model.predict(inputs)
print(output)

###################
# train the model #
###################


# inputs: 
X = [
    [0.1, 0.2],
    [0.2, 0.1],
    [0.1, 0.1],
    [0.1, 0.2],
    [0.1, 0.1]
]
# turn to numpy array
X = np.array(X)

# actual outputs: 
y = [
    [1, 0],
    [0, 1],
    [1, 1], 
    [1, 0], 
    [1, 1]
]
# turn to numpy array
y = np.array(y)


# make the model learn 10 times
for _i in range(10):
    for i in range(len(X)):
        model.train(X[i], y[i], learning_rate=0.01)

# see predictions after training
for i in range(len(X)):
    print("prediction " , i, ": ")
    print("pridiciton: ", model.predict(X[i]))
    print("expected  : ", y[i])


