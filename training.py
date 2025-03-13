#in the last exercise, we optimized weights and biases using the following code:
#predictions = model(X) #forward pass
#MSE = loss(predictions, y) #compute loss
#MSE.backward() #compute gradients
#optimizer.step() #update weights and biases

#We'll turn this code into a training loop that will perform multiple optimizations in a row, attempting to 
#decrease the loss as much as possible

#The only additional line of code we'll need inside the loop is 
#optimizer.zero_grad()
#this line resets the gradients for each iteration. The gradients determine the direction to move the
#weights and biases, and we want to pick a brand new direction each time.

#Here's an example loop. Note that in neural networks, iterations of the training loop are called epochs
#num_epochs = 1000
#for epoch in range(num_epochs):
    #predictions = model(X) #forward pass
    #MSE = loss(predictions, y) #compute loss
    #MSE.backward() #compute gradients
    #optimizer.step() #update weights and biases
    #optimizer.zero_grad() #reset the gradients for the next iteration

#Here's an example of how to print out the epoch number and loss every 100 epochs:

#keep track of the loss values during training
#if (epoch + 1) % 100 == 0:
    #print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

#since epoch starts at 0, the 100th epoch occurs at epoch = 99. So the line if(epoch + 1) % 100 == 0
#checks if the current epoch is the 100th, 200th, and so on

#we call .item() on MSE to print out only the loss value. If we just called MSE on its own, we'd also print
#out the grad_fn

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

apartments_df = pd.read_csv("learn-pytorch/teste.csv")
numerical_features = ['size_sqft', 'bedrooms', 'building_age_years']

#create tensor of input features
X = torch.tensor(apartments_df[numerical_features].values, dtype = torch.float)

#create tensor of targets
y = torch.tensor(apartments_df['rent'].values, dtype = torch.float).view(-1, 1)

#define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

#MSE loss function + optimizer
loss = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X)
    MSE = loss(predictions, y)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    #keep track of the loss during training
    if(epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], MSE Loss: {MSE.item()   }')