import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import log
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
import pandas as pd

train = pd.read_csv('C:\\Users\\Momo\\Desktop\\The_hand_crafted\\ANN\\Train.csv')

y = np.array(train.iloc[:, 14], dtype=float)
X = train.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].values


lr = XGBRegressor(random_state=1)

efs = EFS(lr,
          min_features=4,
          max_features=4,
          scoring='neg_mean_squared_error',
          print_progress=True,
          cv=5)

df_X = pd.DataFrame(X, columns=['ASA','BGN','RMSENG','ZCR','MFCC1','MFCC2','MFCC4','MFCC8','S_Decrease','S_Flatness','S_Flux','S_Rolloff','S_Slope','S_Spread'])
df_X.head()

efs1 = efs.fit(df_X, y)

print('Best MSE score: %.4f' % efs1.best_score_ * (-1))
print('Best subset (indices):', efs1.best_idx_)
print('Best subset (corresponding names):', efs1.best_feature_names_)

