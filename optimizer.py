#to improve on our model's performance, we need to adjust the weights and biases. This is what the optimizer
#algorithm does

#there are many optimization algorithms, one of them is

#Gradient Descent
#Imagine we're on top of a mountain, and our loss function tells us how high up we are. Our goal is to get
#down the mountain, making the loss function as small as possible

#Suppose we can determine how the mountain slopes around us for about a meter in any direction. With that
#information, we might pick where the mountain slopes down the most and move in that direction

#we don't want to move too far at once, because the mountain slope might change as we move. So we'll only
#move a short distance in our chosen direction before pausing and re-evaluating which direction goes
#downhill the most

#This strategy is essentially how gradient descent works. It uses calculus to determine the gradients of the
#loss function. These gradients are the direction signs that indicate which way to adjust the weights and
#biases in order to decrease the loss function

#Learning Rate
#How far to move at each step is called the learning rate
#A lr too high may cause the model to move too quickly and miss the lowest point
#A lr too small may cause the model to learn slowly or get stuck

#The learning rate is an example of hyperparameter - values tuned and tweaked by ML engineers during training
#to improve performance of the model. The process of adjusting hyperparameters is called hyperparameter tuning

#Using Optimizers in PyTorch
#A popular optimizer in pytorch, called Adam, uses gradient descent with a few extra bells and whistles (like 
#adjusting the lr dynamically during training). To use Adam, we'll use the syntax

#import torch.optim as optim
#optimizer = optim.Adam(model.parameters(), lr = 0.01)
#model.parameters() tells Adam what our current weights and biases are
#lr = 0.01 tells Adam to set the learning rate to 0.01

#To apply Adam to a neural network, we need to perform the:

#backward pass: calculate the gradients of the loss function (these determine the "downward" direction)
#step: use the gradients to update the weights and biases

#The syntax is:
#compute loss
#MSE = loss(predictions, y)
#backward pass to determine "downward" direction
#MSE.backward()
#apply the optimizer to update weights and biases
#optimizer.step()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1)
)

#import data
apartments_df = pd.read_csv("learn-pytorch/teste.csv")
numerical_features = ['bedrooms', 'size_sqft', 'building_age_years']
X = torch.tensor(apartments_df[numerical_features].values, dtype = torch.float)
y = torch.tensor(apartments_df[['rent']].values, dtype = torch.float)
#por algum motivo, se não tiver mais um [] em rent, o formato do tensor é diferente do input tensor
#e aparece um aviso no output
#y = torch.tensor(apartments_df['rent'].values, dtype = torch.float)

#forward pass
predictions = model(X)

#define the loss function and compute loss
loss = nn.MSELoss()
MSE = loss(predictions, y)
print("Initial loss is: ", MSE)

optimizer = optim.Adam(model.parameters(), lr = 0.01)

#perform backward pass
MSE.backward()

optimizer.step()

#feed the data through the updated model and compute the new loss
predictions = model(X)
MSE = loss(predictions, y)
print("After optimization, loss is ", MSE)