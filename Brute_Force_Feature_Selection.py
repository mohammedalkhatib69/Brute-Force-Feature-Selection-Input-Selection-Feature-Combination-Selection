### This is an example using brute force feature selection/ feature combination selection for a regression problem 
### this code can be reused for a classification problem, you only need to change the model and scoring criteria

## Strat
# import libraries 
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# ExhaustiveFeatureSelector is responsible for brute force feature selection 
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

# import training dataset using pandas
train = pd.read_csv('C:\\Users\\Momo\\Desktop\\Train.csv')


# define your inputs and output, in this case, we have 14 inputs and 1 output, all numeric
X = np.array(train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]], dtype=float)
y = np.array(train.iloc[:, 14], dtype=float)

# define the model, any model can be used 
lr = XGBRegressor(random_state=1)

# Deifne the ExhaustiveFeatureSelector paramters, in this case there are 14 ponteital features/ inputs 
# So we try to map through all of them and select the best feature combination
# If you are only interested in finding the best feature combination of 3 features only 
# Then you need to define min_features =3 and max features =3 
# Remember that always min features is either = or < to the max features
# socring is another important parameter. It defines the selection criteria. you can use R2, MAE, MSE or define your own score
# in this case, we used 'neg_mean_squared_error because the search will only output the value of the height, so remember -1 is larger than -2
# deifen cross-validation (CV) the default is 5


efs = EFS(lr,
          min_features=1,
          max_features=14,
          scoring='neg_mean_squared_error',
          print_progress=True,
          cv=5)

efs1 = efs.fit(df_X, y)

print('Best MSE score: %.4f' % efs1.best_score_ * (-1))
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)

## The end ##
