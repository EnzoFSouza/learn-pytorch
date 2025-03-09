#suppose we want to create the following neural network
#input layer: two nodes
#hidden layer: three nodes with ReLU activation
#output layer: one node
#nn.Linear() defining the layers and performing the linear calculations (weighted inputs + bias)
#nn.ReLU() for applying the activation function
#Here's how we would build the example network
#model = nn.Sequential(
#    nn.Linear(2, 3), connects the 2 input nodes to the 3 hidden nodes using standard weights-and-bias linear calculation
#    nn.ReLU(), applies the ReLU activation function to that linear computation
#    nn.Linear(3, 1) connects the ReLU output from the three hidden nodes to the one output node (using linear calculation)
#)

#Initially, each instance of nn.Linear() will use randomly generated weights and bias. In the training process, the neural network will
#update these to improve its predictions

#Running feedforward
#To run through the feedforward process with our Sequential model, we need to pass a tensor as input to model
#Typically our input tensors will be two-dimensional, consisting of
#rows: individual examples (in this case, each row is an apartment)
#columns: indiviual features (in this case, each column is an apartment property, like size)

#Suppose the two input nodes in our model correspond to building age and number of bedrooms
#Let's create an input tensor with a couple of apartments and run the feedforward process
#typically input tensors are stored in a variable named X. The input tensor is usually capitalized to indicate that it is a 
#two-dimensional matrix of rows and columns
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#nn.Sequential randomly generates weights and biases. Set a random seed so the INITIAL weights and biases are the same
#every time we run the same code
torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(2, 3),
    nn.ReLU(),
    nn.Linear(3, 1)
)

#create apartment data
apts = np.array(
    [[100, 3], #100 years old, 3 bedrooms
    [50, 4]]) #50 years old, 4 bedrooms

#convert to a tensor
X = torch.tensor(apts, dtype = torch.float)

#run feedforward
#model(X)

resultado = model(X)
print(resultado)
#input row [100, 3] corresponds to output/prediction of -23.0715
#input row [50, 4] corresponds to output/prediction of -11.8710
#as predictions these make no sense, but we haven't trained the model at all, so that isn't surprising

#We can interpret the output by associating each row of output to the same row of the input tensor
#Exercise 1
#Use nn.Sequential to create a neural network model with 
#input layer: three nodes
#hidden layer: eight nodes
#output layer: one node
#Assign your network to the variable 'model'
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)
print(model) #show model details

#Exercise 2
#add a second hidden layer with four nodes and nn.Sigmoid as the activation function
model = nn.Sequential(
    nn.Linear(3, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.Sigmoid(),
    nn.Linear(4, 1)
)

print(model)

#Exercise 3
#Let's create a model and feedforward data

#load pandas dataframe
apartments_df = pd.read_csv("learn-pytorch/teste.csv")

#create a numpy array of the numeric columns
apartments_numpy = apartments_df[['size_sqft', 'bedrooms', 'building_age_years']].values
#print(apartments_numpy)

#convert to an input tensor
X = torch.tensor(apartments_numpy, dtype = torch.float32)

print(X[:5]) #preview the first five apartments

#Run the feedforward process of model on X. Assign the result to the variable 'predicted_rent'

#define the neural network
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

predicted_rent = model(X)

#show output
print(predicted_rent[:5])