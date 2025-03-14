#decreasing the loss during training only tells us that the model is becoming more accurate on the data 
#it is learning from
#We also need to evaluate the model on new data and save the best model for future use

#ML engineers usually only train models on a portion of the original dataset (called the training data) and
#then evaluate it on the remainder (called the testing data). This process is called train-test split
#We'll use the python library scikit-learn to perform the split. The syntax is : 
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y,
#   train_size = 0.80,
#   test_size = 0.20,
#   random_state = 2)

#where:
#X contains all of our input features
#y contains the targets we are trying to predict
#train_size = .8 tells scikit-learn to use 80% of the data for training
#test_size = .2 tells scikit-learn to use the remaining 20% for testing 
#random_state = 2 sets a random state, so that we create the same train-test split each time we run the code

#This produces two sets of data:
#data for training the model: X_train (input features) and y_train (corresponding targets)
#data for testing the model: X_test (input features) and y_test (corresponding targets)

#Evaluation
#To evaluate our model on new data, we will use our testing data with the following syntax
#model.eval() #sets the model to evaluation mode
#with torch.no_grad(): #turns off gradient calculations (which we don't need outside of training)
#   predictions = model(X_test) #run feedforward to predict the targets for the test dataset
#   test_MSE = loss(predictions, y_test) #compute the MSE loss between predicted targets and the actual targets

#Saving and Loading models
#save the neural network to a specified path
#torch.save(model, 'model.pth')

#load the saved neural network from the specified path
#loaded_model = torch.load('model.pth')

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

apartments_df = pd.read_csv("learn-pytorch/teste.csv")
numerical_features = ["bedrooms", "size_sqft", "building_age_years"]

#create tensor of input features
X = torch.tensor(apartments_df[numerical_features].values, dtype = torch.float)

#create tensor of targets
y = torch.tensor(apartments_df['rent'].values, dtype = torch.float).view(-1, 1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size = 0.7,
    test_size = 0.3,
    random_state = 2)

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

#training
num_epochs = 1000
for epoch in range(num_epochs):
    predictions = model(X_train)
    MSE = loss(predictions, y_train)
    MSE.backward()
    optimizer.step()
    optimizer.zero_grad()

    #keep track of the loss during training
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], MSE Loss: {MSE.item()}')

#torch.save(model, 'model.pth')

loaded_model = torch.load('learn-pytorch/model.pth')
loaded_model.eval()
with torch.no_grad():
    predictions = loaded_model(X_test)
    test_MSE = loss(predictions, y_test)

#show output
print(f'Test MSE is {test_MSE.item()}')
print(f'Test Root MSE is {test_MSE.item()**(1/2)}')

#plot our model's predictions against the actual targets, for the test dataset
import matplotlib.pyplot as plt


plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, label='Predictions', alpha=0.5, color='blue')

plt.xlabel('Actual Values (y_test)')
plt.ylabel('Predicted Values')

plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='gray', linewidth=2,
         label="Actual Rent")
plt.legend()
plt.title('StreetEasy Dataset - Predictions vs Actual Values')
plt.show()