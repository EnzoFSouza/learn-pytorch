import torch
import pandas as pd
import numpy as np
#Build a neural network to predict the rent of an apartment

#tensors: building locks of neural networks in pytorch
#tensors act as storage containers for numerical data

#define the apartment's rent
rent = 2500

#torch.tensor(numerical_data, data type)
#numerical data can be numpy array, python list, python numeric variable

#convert to an integer tensor
rent_tensor = torch.tensor(
    rent,
    dtype = torch.int
)

#show output
print(rent_tensor)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

#creating numpy array with rent, size_sqft, age
apt_array = np.array([2550, 750, 3.5])

#convert to a tensor of floats
apt_tensor = torch.tensor(
    apt_array,
    dtype = torch.float
)

print(apt_tensor)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#if data start out as a pandas dataframe, we need to use the .values attributte of the df to retrieve data
#as a numpy array

#convert a dataframe named df to a pytorch tensor
#torch.tensor(
#    df.values,
#    dtype = torch.float
#)

#working with individual columns of a df can cause problems due to dimension
#one solution -> make sure the individual column is a full dataframe by usinf two brackets to select it
#torch.tensor(
#    df.[['column']].values,
#    dtype = torch.float
#)

#another is to use .view(-1, 1) to automatically adjust dimensions
#torch.tensor(
#    df['column'].values,
#    dtype = torch.float
#).view(-1, 1)

#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
#import dataset using pandas
apartments_df = pd.read_csv("teste.csv")
#print(apartments_df)
#select rent, size and age columns
apartments_df = apartments_df[["rent", "size_sqft", "building_age_years"]]
#convert dataframe to tensor

apartments_tensor = torch.tensor(
    apartments_df.values,
    dtype = torch.float
)

print(apartments_tensor)
