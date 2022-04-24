'''
Dharma Hoy
04/23/22
Physics 305 Creative Project
Get Data

This program loads the data into a data frame, normalizes it,
then creates a csv of the data so it is easily imported into 
other programs in this project.
The data used is from kaggle. It consists of observations of 
space taken by the Sloan Digital Sky Survey and can be found 
at the link below.
https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17
'''
import pandas as pd

# read data
stars = pd.read_csv("data/star_classification.csv")

# normalized the data 
colToNorm = ['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'run_ID', \
'rerun_ID', 'cam_col', 'filed_ID', 'spec_obj_ID', 'redshift', 'plate', \
'MJD', 'fiber_ID']

stars[colToNorm] = stars[colToNorm].apply(lambda x: x-x.min())/(x.max()-x.min())

# write a csv file of the normalized data
stars.to_csv("data/stars.csv")
