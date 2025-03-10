import numpy as np
import pandas as pd

import torch
import torch.nn as nn

torch.manual_seed(42)

#need to create non-sequential types of neural networks using Object-oriented programming (OOP)
#using OOP gives more control over neural networks
#With Sequential, we can only feed data from one layer to th next
#Building logic to skip certain layers or to loop others can improve the neural network performance. For that, we need OOP

#Building a familiar Sequential network with OOP
#Create the NN_Regression class
#model = nn.Sequential(<inputs>)
#in this syntax, nn.Sequential refers to a type of neural network (sequential neural network). Types of things in OOP are called CLASSES
#the variable 'model' is a specific Sequential neural network, called an INSTANCE of the class
#we can create our own classes (or types) of PyTorch neural networks using 

#Create the NN_Regression class
class NN_Regression(nn.Module):
    #Initialize the network components
    #We need to initialize all the layers and activation functions we plan to use - "gathering all the ingredients"
    def __init__(self):
        super(NN_Regression, self).__init__()
        #Initialize the layers
        self.layer1 = nn.Linear(3, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 4)
        self.layer4 = nn.Linear(4, 1)

        #Initialize activation functions
        self.relu = nn.ReLU()
        
        #The self syntax just allows us to reference these variables, like 'self.layer1', in the next section of defining the class
    
    #Define the forward pass
    #now that we've gathered all the ingredients, we need to describe how to combine them in order to perform the feedforward operation
    #create a 'forward' method that dictates the flow of how an input data tensor 'x' is passed from layer to layer through the network
    def forward(self, x):
        #define the forward pass
        x = self.layer1(x) #takes the input tensor 'x' and passes it through the input layer to the next hidden layer
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.layer3(x)
        x = self.relu(x)
        x = self.layer4(x)
        return x

#We now have a new type of PyTorch neural network called 'NN_Regression'
#Instantiate the model
model = NN_Regression() #create an instance of an 'NN_Regression' neural network

#Exercise 1
#The output from this network should be the same as the sequential network from the last exercise

#create an input tensor
apartments_df = pd.read_csv("learn-pytorch/teste.csv")
numerical_features = ['size_sqft', 'bedrooms', 'building_age_years']
X = torch.tensor(apartments_df[numerical_features].values, dtype = torch.float)

#feedforward to predict rent
predicted_rents = model(X)

print(predicted_rents[:5])

class OneHidden(nn.Module):
    #able to pass num of hidden nodes to the class
    def __init__(self, numHiddenNodes):
        super(OneHidden, self).__init__()
        self.layer1 = nn.Linear(2, numHiddenNodes)
        self.layer2 = nn.Linear(numHiddenNodes, 1)

        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = OneHidden(10)

input_tensor = torch.tensor([3, 4.5], dtype = torch.float32)

#run feedforward
predictions = model(input_tensor)

print(predictions)
