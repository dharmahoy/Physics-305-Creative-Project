'''
Dharma Hoy
04/23/22
Physics 305 Creative Project
Get Data

This program loads the data into a data frame, normalizes it,
then creates a csv of the data so it is easily imported into 
other programs in this project. This program also narrows down
the columns in the data set. The columns that aren't used are
the ones that are ID values that are not object ID.
The data used is from kaggle. It consists of observations of 
space taken by the Sloan Digital Sky Survey and can be found 
at the link below.
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
'''
import pandas as pd

# read data
stars = pd.read_csv("data/star_classification.csv")

# narrow down columns
stars = pd.DataFrame(stars[['obj_ID', 'alpha', 'delta', 'u', 'g', 'r', 'i', 'z', \
'class', 'redshift', 'MJD']])

# normalized the data 
colToNorm = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'redshift', 'MJD']

for col in colToNorm:
  stars[col] = ((stars[col]-stars[col].min()) / (stars[col].max()-stars[col].min()))

# write a csv file of the normalized data
stars.to_csv("data/normalized_star_classification.csv")
