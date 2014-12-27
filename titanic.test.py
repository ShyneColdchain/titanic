import pandas as pd
import numpy as np

data = pd.read_csv('train.csv', dtype = str)
survive = data['Survived']
data = data.drop('Survived', 1)

for colname in list(data.columns.values):
	data[colname] = pd.Categorical.from_array(data[colname]).codes