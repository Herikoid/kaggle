# data analysis and wrangling
import pandas as pd
import numpy as np

# converter for categorical features
from sklearn.feature_extraction import DictVectorizer

# machine learning
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from statistics import mean

# time measurement
import datetime

# chdir
import os

os.chdir('/home/mikhail/PycharmProjects/kaggle/HousePrices')

# Data analysis

# reading data
data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')

# getting a general look at features
print(data_train.info())
print('_' * 40)
print(data_test.info())

# getting summary about features
print(data_train.describe())
print(data_train.describe(include=['O']))

# saving 'Id' column for submission
Id = data_test['Id']

# Data preprocessing

# dropping features
data_train = data_train.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
data_test = data_test.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)

# answers extraction
y_train = data_train['SalePrice']

# separation of numeric and categorical features
num_features = list(data_test.describe())
cat_features = list(data_test.describe(include=['O']))

# numeric features extraction
X_train_num = data_train[num_features]
X_test_num = data_test[num_features]

# num NA processing
for name in num_features:
    name_mean = X_train_num[name].mean()
    X_train_num[name].fillna(name_mean, inplace=True)
    X_test_num[name].fillna(name_mean, inplace=True)

# Categorical features extraction
X_train_cat = data_train[cat_features]
X_test_cat = data_test[cat_features]

# cat NA processing
X_train_cat.fillna('nan', inplace=True)
X_test_cat.fillna('nan', inplace=True)

# creation of binary features from categorical
enc = DictVectorizer()
X_train_cat = enc.fit_transform(X_train_cat.to_dict('records'))
X_test_cat = enc.transform(X_test_cat.to_dict('records'))

# features reuniting
X_train = np.hstack([X_train_num, X_train_cat.toarray()])
X_test = np.hstack([X_test_num, X_test_cat.toarray()])

# Machine learning

start_time = datetime.datetime.now()

# finding best number of trees for gradient boosting using cross-validation
score = []
for k in range(201, 220, 1):
    start_time_temp = datetime.datetime.now()
    cv_score = []
    rgr = GradientBoostingRegressor(n_estimators=k, random_state=1)
    cv = KFold(n_splits=5, shuffle=True, random_state=1)
    for train_index, test_index in cv.split(X_train, y_train):
        rgr.fit(X_train[train_index], y_train.values[train_index])
        cv_score.append(
                mean_squared_error(
                    y_train.values[test_index],
                    rgr.predict(X_train[test_index])
                )
        )
    score.append([mean(cv_score), k, datetime.datetime.now() - start_time_temp])

# fitting classifier
rgr = GradientBoostingRegressor(n_estimators=min(score)[1], random_state=1).fit(X_train, y_train)

# getting prediction for test sample
y_pred = rgr.predict(X_test)

print('Time passed:', datetime.datetime.now() - start_time)

# Output

# writing prediction to the file
prediction = pd.DataFrame({
        'Id': Id,
        'SalePrice': y_pred
    })
prediction.to_csv('output/gb_ca.csv', index=False)
