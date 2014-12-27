import pandas as pd
import numpy as np
import inspect

from sklearn.ensemble import GradientBoostingClassifier 

def get_lineage(tree, feature_names):
     left      = tree.children_left
     right     = tree.children_right
     threshold = tree.threshold
     features  = [feature_names[i] for i in tree.feature]

     # get ids of child nodes
     idx = np.argwhere(left == -1)[:,0]     

     def recurse(left, right, child, lineage=None):          
          if lineage is None:
               lineage = [child]
          if child in left:
               parent = np.where(left == child)[0].item()
               split = 'l'
          else:
               parent = np.where(right == child)[0].item()
               split = 'r'

          lineage.append((parent, split, threshold[parent], features[parent]))

          if parent == 0:
               lineage.reverse()
               return lineage
          else:
               return recurse(left, right, parent, lineage)

     for child in idx:
          for node in recurse(left, right, child):
               print node
               
def main():
    data = pd.read_csv('train.csv', dtype = str)
    # .values = convert to np
    survive = data['Survived'].astype(int).values
    data = data.drop('Survived', 1)
    feature_name = list(data.columns.values)

    for colname in list(data.columns.values):
    	data[colname] = pd.Categorical.from_array(data[colname]).codes

    # convert to np    
    data = data.values
 
    # gradient boosting     
    gbt = GradientBoostingClassifier(verbose = 1)
    gbt.fit(data, survive)
    gbt.predict(data[0, :]) # predict first row
    
    tree = gbt.estimators_[0][0].tree_

    get_lineage(tree, feature_name)

    ################################################
    # inspect elements 
    
    #inspect.getmembers(gbt.estimators_[0][0].tree_)
    # or
    #dir(gbt.estimators_[0][0].tree_)

    #dir(gbt.estimators_[0][0].tree_.apply)  # inside .apply
    ################################################
    
               
if __name__ == '__main__':
    main()
    

