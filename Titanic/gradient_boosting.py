# data analysis and wrangling
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import os

# machine learning
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

# chdir
os.chdir('/home/mikhail/PycharmProjects/kaggle/Titanic/data')


# Reading data
data_train = pd.read_csv('train.csv', index_col='PassengerId')
data_test = pd.read_csv('test.csv', index_col='PassengerId')


# Data preprocessing

# Answers extraction
y_train = data_train.Survived

# Numeric features extraction
X_train_num = data_train.drop(['Survived', 'Name', 'Ticket', 'Cabin',
                               'Embarked'], axis=1)
X_test_num = data_test.drop(['Name', 'Ticket', 'Cabin', 'Embarked'],
                            axis=1)

# NA processing
age_mean = X_train_num['Age'].mean()
X_train_num['Age'].fillna(age_mean, inplace=True)
X_test_num['Age'].fillna(age_mean, inplace=True)
X_test_num['Fare'].fillna(X_train_num['Fare'].mean(), inplace=True)

# Transforming Sex feature to numeric
X_train_num['Sex'] = X_train_num['Sex'].apply(lambda x: 1 if x == 'male'
                                              else 0)
X_test_num['Sex'] = X_test_num['Sex'].apply(lambda x: 1 if x == 'male'
                                            else 0)

# Text features extraction
X_train_text = data_train['Name'].apply(lambda x: x.lower())
X_test_text = data_test['Name'].apply(lambda x: x.lower())

# NA processing
X_train_text.fillna('nan', inplace=True)
X_test_text.fillna('nan', inplace=True)

# Extra characters deleting
X_train_text = X_train_text.replace('[^a-zA-Z0-9]', ' ', regex=True)
X_test_text = X_test_text.replace('[^a-zA-Z0-9]', ' ', regex=True)

# Tf-idf application to text features
vectorizer = TfidfVectorizer()
X_train_text = vectorizer.fit_transform(X_train_text)
X_test_text = vectorizer.transform(X_test_text)

# Categorical features extraction
X_train_categ = data_train[['Cabin', 'Embarked']]
X_test_categ = data_test[['Cabin', 'Embarked']]
# NA processing
X_train_categ.fillna('nan', inplace=True)
X_test_categ.fillna('nan', inplace=True)

# Creation of binary features from categorical
enc = DictVectorizer()
X_train_categ = enc.fit_transform(X_train_categ.to_dict('records'))
X_test_categ = enc.transform(X_test_categ.to_dict('records'))

# Features reuniting
X_train = np.hstack([X_train_num, X_train_categ.toarray(),
                     X_train_text.toarray()])
X_test = np.hstack([X_test_num, X_test_categ.toarray(), X_test_text.toarray()])


# Finding best number of trees for gradient boosting using cross-validation
score = []
for k in range(100, 200, 10):
    clf = GradientBoostingClassifier(n_estimators=100, random_state=1)
    cv_score = cross_val_score(clf, X_train, y_train, scoring='accuracy',
                               cv=KFold(n_splits=5, random_state=1))
    score.append([cv_score.mean(), k, 3])


# Fitting classifier
clf = GradientBoostingClassifier(n_estimators=max(score)[1], random_state=1
                                 ).fit(X_train, y_train)


# Getting prediction for test sample
pred = clf.predict(X_test)

# Writing prediction to the file
with open('prediction_gb.csv', 'w') as f:
    f.write('PassengerId' +
            pd.DataFrame(pred, index=range(892, 1310),
                         columns=['Survived']).to_csv())
    f.close()
