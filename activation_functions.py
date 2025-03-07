#One of the ways neural networks move beyond linear regression is by
#incorporating non-linear activation functions
#these functions allow a neural network to model non-linear relationships within a dataset
#activation functions take the linear equation as input, and modify it to introduce non-linearity
#Process with an activation function:
#   receive the weighted inputs
#   add them up (to produce the same linear equation [as the output node])
#   apply an activation function

#ReLU activation function - one of the most common
#if a number is negative, ReLU returns 0
#If a number is positive, ReLU returns the number with no changes
#ReLU(-1) = 0, since -1 is negative
#ReLU(.5) = .5, since .5 not negative

#if the weighted inputs coming into a ReLU node are 3 and -4, the output would be calculated by:
#receive the weighted inputs: 3 and -4
#add them together: 3+(-4) = 3-4 = -1
#apply ReLU: ReLU(-1) = 0
#output = 0

#Other activation functions
#sigmoid activation function only outputs values between 0 and 1, and is common in classification problems

#Use icons to indicate the activation function in diagrams

#ReLU(-3) = 0
#ReLU(0) = 0
#ReLU(3) = 3

#Suppose the weighted inputs are -3.5 and 3
#Compute the output of the node in two ways:
#1. Linear output/No activation
#2. ReLU activation

def ReLU(x):
    return max(0, x)

linear_output = -3.5 + 3
ReLU_output = ReLU(linear_output)

print("Linear node output: ", linear_output)
print("ReLU node output: ", ReLU_output)

