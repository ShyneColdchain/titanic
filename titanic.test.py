import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier 

data = pd.read_csv('train.csv', dtype = str)
# .values = convert to np
survive = data['Survived'].astype(int).values
data = data.drop('Survived', 1)

for colname in list(data.columns.values):
	data[colname] = pd.Categorical.from_array(data[colname]).codes

# convert to np    
data = data.values
 
# gradient boosting     
gbt = GradientBoostingClassifier(verbose = 1)
gbt.fit(data, survive)