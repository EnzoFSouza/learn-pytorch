#regression problem in machine learning: trying to predict a target numeric value
#linear regression: most basic kind of regression
#linear regression models use linear equation to make predictions y = mx + b
#rent = 2.5sz_sqft + 1000
#rent: output or target
#sz_sqft: input_feature
#2.5: weight associated to sz_sqft
#1000: bias
#More than one input feature:
#rent = 2.5sz_sqft - 1.5age + 1000
#weighted input feature: each input feature is multiplied by its own weight

#considering rent = 3sz_sqft + 500 and sz_sqft = 500
predicted_rent = 3 * 500 + 500

#considering rent = 3sz_sqft + 10bedrooms + 250
bedroom_weight = 10
bias = 250