#rent = 2.5sz_sqft - 1.5age + 1000
#transform this equation into a neural network structure called Perceptron
#Perceptron is a type of network structure consisting of nodes (circles, nós) connected to each other by edges (arrows)
#The nodes appear in vertical layers, connected from left to right
#What makes a network a perceptron is that it has one set of input nodes leading to a single output node

#Define the inputs
size_sqft = 500.0
age = 10.0
bias = 1.0

#The inputs flow through the edges, receiving weights
weighted_size = 2.5 * size_sqft
weighted_age = -1.5 * age
weighted_bias = 1000 * bias

#The output node adds the weighted inputs
weighted_sum = weighted_size + weighted_age + weighted_bias

#Generate prediction
print("Predicted rent: ", weighted_sum)

#Add an additional input feature "bedrooms" to our linear regression perceptron which
#corresponds to the number of bedrooms in the apartment
#rent = 3sz_sqft - 2.3age + 100bedrooms + 500

#Define the inputs
size_sqft = 1250.0
age = 15.0
bedrooms = 2.0
bias = 1.0

#The inputs flow through the edges, receiving weights
weighted_size = 3 * size_sqft
weighted_age = -2.3 * age
weighted_bedrooms = 100 * bedrooms
weighted_bias = 500 * bias

#The output node adds the weighted inputs
weighted_sum = weighted_size + weighted_age + weighted_bedrooms + weighted_bias

#Generate prediction
print("Predicted rent: ", weighted_sum)