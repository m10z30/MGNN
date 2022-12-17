import numpy as np
import random as rn
import pickle


# layer class 
class Layer:
    # init function
    def __init__(self, n_inputs, n_neurons, activation=None) -> None:
        # initializing weights, biases etc...
        self.weights = (np.random.rand(n_inputs, n_neurons) * 6) - 3 
        self.biases = np.zeros((1, n_neurons))
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = "softmax" if activation == None else activation
    

    def copy(self): # copy the layer
        copy_weights = np.copy(self.weights)
        copy_biases = np.copy(self.biases)
        copy_layer = Layer(self.n_inputs, self.n_neurons, self.activation)
        copy_layer.insert_values(copy_weights, copy_biases)
        return copy_layer
    
    def insert_values(self, new_weights, new_biases):
        self.weights = new_weights
        self.biases = new_biases

    def forward(self, inputs): # getting the output of the layer
        self.outputs = np.dot(inputs, self.weights) + self.biases
        if self.activation == "sigmoid":
            self.outputs = self.sigmoid(self.outputs)
        elif self.activation == "softmax":
            self.outputs = self.softmax(self.outputs)
        elif self.activation == "ReLU":
            self.outputs = self.ReLU(self.outputs)
        elif self.activation == "tanh":
            self.outputs = self.tanh(self.outputs)
        else:
            raise("no such activation function")
        return self.outputs
    
    def backward(self, inputs, index, layers,y ,learning_rate): # making the layer learn
        # first getting the output
        output = self.forward(inputs)

        # calculating the learning rate & modify the weights & biases of the layer

        if index == 0:
            inps = np.zeros((len(inputs), 1))
            for i in range(len(inputs)):
                inps[i][0] = inputs[i]
            inputs = inps
            

        if index == len(layers) - 1:
            error = y - output
            delta = error * output * (1 - output) * learning_rate
            self.weights += delta * inputs.T
            self.biases += delta 
            return self.weights , error
        else:
            next_weights, next_error = layers[index+1].backward(output, index+1, layers, y, learning_rate)
            
            
            error = np.dot(next_weights , next_error.T)
            error = error.T
            delta = error * output * (1 - output) * learning_rate
            self.weights += delta * inputs
            self.biases += delta

            return self.weights , error
        


    def mutate(self, mutation_rate): # mutating the weights and biases randomly
        for i in range(self.n_neurons):
            self.biases[0][i] += rn.uniform(-mutation_rate, mutation_rate)
            for j in range(self.n_inputs):
                self.weights[j][i] += rn.uniform(-mutation_rate, mutation_rate)
    
    def sigmoid(self,x): # sigmoid function
        return 1 / (1 + np.exp(-x))
    
    def ReLU(self, x): # ReLU function
        return np.maximum(0, x)

    def softmax(self, x): # softmax function
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities
    
    def tanh(self, x): # tanh function
        return 2*self.sigmoid(2*x)-1



# this is the Neural Network model 
# it contains the layers
class Model:
    def __init__(self, shape=None, mutation_rate=0.01, activations=None) -> None:
        # the shape is a tuple or any iterable object
        # example: shape = (2, 3, 4, 3, 2) 
        # this is the shape of the neural network

        # the mutation rate is the rate at which the neural network can mutate 

        # activations is a tuple or any iterable object
        # it has to be of length len(shape) - 1 
        # it must contain the list of strings of the activation functions for each layer
        # available activations: ("sigmoid", "ReLU", "softmax", "tanh")


        self.mutation_rate = mutation_rate
        self.shape = shape
        self.layers = []
        self.activations = activations
        if activations != None:
            if len(shape) - 1 != len(activations):
                raise("the number of layers diffrent from activations")
        prev_i = None
        index = 0
        for i in shape:
            if prev_i == None:
                prev_i = i
            else:
                if activations == None:
                    new_layer = Layer(prev_i, i)
                    self.layers.append(new_layer)
                    prev_i = i
                else:
                    new_layer = Layer(prev_i, i, activations[index])
                    self.layers.append(new_layer)
                    prev_i = i
                    index += 1
                
        
    def get_shape(self): # getting the shape of the NN
        return self.shape
    
    def predict(self, inputs): # getting the output of the NN
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def mutate(self): # mutate the NN
        for layer in self.layers:
            layer.mutate(self.mutation_rate)
    
    def copy(self): # returns a copy of the current Model
        copy_model = Model(self.shape, self.mutation_rate, self.activations)
        copy_layers = []
        for layer in self.layers:
            copy_layers.append(layer.copy())
        copy_model.insert_layers(copy_layers)
        return copy_model
    
    def insert_layers(self, layers): # inserts an array of layers
        self.layers = layers
    

    def train(self, X, y, learning_rate=0.01): # train the Model
        # X: is the input
        # y: is the correct values
        # learning_rate: by default 0.01
        self.layers[0].backward(X, 0, self.layers, y, learning_rate)
        # output = self.predict(X)
        # error = output - y
        # last_output = self.predict(X)
        # for i in range(len(self.layers)-1, 0, -1):
        #     if i == 0:
        #         outputs = self.layers[i].forward(last_output)
        #         self.layers[i].backward(last_output, outputs, learning_rate)
        #         last_output = outputs
                
        #     elif i == len(self.layers)-1:
        #         outputs = self.layers[i].forward(last_output)
        #         self.layers[i].backward(last_output, outputs, learning_rate)
        #         last_output = outputs

        #     else:
        #         outputs = self.layers[i].forward(last_output)
        #         self.layers[i].backward(last_output, outputs, learning_rate)
        #         last_output = outputs
            
            # weightsT = self.layers[i].weights.T
            # error = weightsT * output 
            # greident = output * (1 - output)
            # output *= error
            # output *= learning_rate


            



    def save(self, name): # saves the model as a pickle file
        mdl = {}
        mdl['layers'] = {}
            
        for i in range(len(self.layers)):
            layer = {}
            layer['weights'] = self.layers[i].weights
            layer['biases'] = self.layers[i].biases
            layer['n_inputs'] = self.layers[i].n_inputs
            layer['n_neurons'] = self.layers[i].n_neurons
            layer['activation'] = self.layers[i].activation

            mdl['layers']['layer'+str(i)] = layer
            
        mdl['mutation_rate'] = self.mutation_rate
        mdl['shape'] = self.shape
        mdl['activations'] = self.activations        
        
        pickle_out = open(str(name)+'.pickle', 'wb')
        pickle.dump(mdl, pickle_out)
        pickle_out.close()    
    

    def load(self, name): # loads the model from a pickle value
        pickle_in = open(str(name)+".pickle", 'rb')
        mdl = pickle.load(pickle_in)
        pickle_in.close()
        
        self.mutation_rate = mdl['mutation_rate']
        self.shape = mdl['shape']
        self.activations = mdl['activations']
        self.layers = []


        for i in range(len(mdl['layers'])):
            weights = mdl['layers']['layer'+str(i)]['weights']
            biases = mdl['layers']['layer'+str(i)]['biases']
            n_inputs = mdl['layers']['layer'+str(i)]['n_inputs']
            n_neurons = mdl['layers']['layer'+str(i)]['n_neurons']
            activation = mdl['layers']['layer'+str(i)]['activation']

            

            layer = Layer(2, 2, None)
            layer.activation = activation
            layer.n_inputs = n_inputs
            layer.n_neurons = n_neurons
            layer.insert_values(weights, biases)
            self.layers.append(layer)
            











