#we predicted the rent of some apartments by feeding them through neural networks. The networks were untrained
#so the predictions weren't good
#Let's start improving those predictions
#To make better predictions, we first need to know how bad our current predictions are and how to track 
#improvement as we train our model
#This is the role of the loss function
#The loss function is a mathematical formula used to measure the error (also known as loss values) between 
#the model predictions and the actual target values (sometimes called labels) the model is trying to predict
#Difference
#suppose we run a neural network on an apartment with $1000/mo rent but the network predicts $500
#the simplest way to measure the loss in this case is to calculate the difference: 500 - 1000 = -500
#for just one data point, the difference seems reasonable. But imagine we have another apartment with
#$1500/mo rent and the model overestimates $2000. The difference in this case is 2000 - 1500 = 500
#with one loss of 500 and another of -500, the average loss is 0. But that doesn't make sense, since the model
#isn't accurate
#The problem is that the negative loss canceled out the positive loss. To fix this, we need to force
#the difference to be always positive

#Mean Squared Error (MSE)
#MSE makes differences positive by squaring them. To calculate MSE on our two example apartments, we would
#calculate the differences: 500 and -500
#square both: 500**2 and (-500)**2
#take the average: (500**2 + (-500)**2)/2 = 250000
#A loss of 250000 seems high, but remember that we've squared the differences. To help interpret MSE, 
#we'll sometimes take the square root of MSE:
#sqrt(250000) = 500
#An average loss of 500 makes sense in this case

#MSE in PyTorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

#To use PyTorch's implementation of MSE:
loss = nn.MSELoss()

#Now that we've instantiated 'loss', we can compute the mean squared error by passing two inputs:
#the predicted values
#the actual target values

#Just as we use 'X' to stand for input features in a neural network, it is common in machine learning to use
#the variable 'y' to stand for the target values. In this case, our target isn't two-dimensional, so we use
#lowercase 'y'

#Let's calculated MSE for our two example apartments
predictions = torch.tensor([500, 2000], dtype = torch.float)
y = torch.tensor([1000, 1500], dtype = torch.float)
print(loss(predictions, y))

#It is important to select the right loss function, sometimes it is worth experimenting with a few to see
#how each behaves
#For example, the squaring process in MSE emphasizes the largest differences between predicted and target
#values. Sometimes, this is helpful, but in other cases it can lead to overfitting. In those cases, instead
#of squaring differences we might choose to take the absolute value to produce positive values (this
#is called the Mean Absolute Error)

predictions = torch.tensor([-6.9229, -29.8163, -16.0748, -13.2427, -14.1096], dtype = torch.float)
y = torch.tensor([2550, 11500, 3000, 4500, 4795], dtype = torch.float) #target values

loss = nn.MSELoss()
MSE = loss(predictions, y)
print("MSE Loss: ", MSE)

#Compute the square root of the MSE tensor. Assign the result to RMSE (root mean squared error)
RMSE = MSE ** (1/2)
print("RMSE Loss: ", RMSE) #tensor(6197.8726)

#looks like this model's error for rent in dollars is around $6200. We can improve this in training
