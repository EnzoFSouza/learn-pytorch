#to really model arbitrarily complex datasets, we need a more complex multi-layer network structure
#we also need to train the network on our data, so it can learn the complex relationships within the dataset

#Hidden Layers: layers of nodes introduced between the input and output layer

#input data flows through the network from the input layer, to the hidden layers, and on to the output
#each node in a hidden layer:
# 1. receives weighted inputs from the nodes in the prior layers
# 2. adds up all those weighted inputs with a bias term
# 3. optionally applies an activation function to the weighted sum
# 4. sends the result to every node in the next layer
 
#Remember that every weighted sum also includes a bias term
 
#Training process
#the training process has roughly four steps
# 1. Forward pass / feedforward: we feed input data through the
# network from layer to layer and calculate the final output value
# 2. Loss: we measure how close (or far) the network's predictions are to the actual values
# 3. Backward pass / backpropagation: we apply an optimization algorithm to go back and update the weights and
# biases of the network, to try to improve the network's performance
# 4. Iterate: we repeat this process over and over, checking each time if the loss (or error) is going down