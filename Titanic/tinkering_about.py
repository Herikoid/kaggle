# data analysis and wrangling
import pandas as pd
import numpy as np
# import random as rnd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.feature_extraction import DictVectorizer

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

# chdir and time
import os
import datetime

os.chdir('/home/mikhail/PycharmProjects/kaggle/Titanic/data')

# Data analysis

# Reading data sets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]
# Wrangling data

# Cabin processing
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].fillna('Unknown')
    dataset['Deck'] = dataset['Cabin'].str.get(0)
combine = [train_df, test_df]

# Ticket processing
Ticket_Count = dict(pd.concat([train_df['Ticket'], test_df['Ticket']]).value_counts())


def Ticket_Label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 8)) | (s == 1):
        return 1
    elif s > 8:
        return 0


for dataset in combine:
    dataset['TicketGroup'] = dataset['Ticket'].map(Ticket_Count)
    dataset['TicketGroup'] = dataset['TicketGroup'].apply(Ticket_Label)

# Dropping features
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Creating new feature extracting from existing
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Uniting rare titles and replacing some titles with a more common name
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Dropping features
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# Converting a categorical feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Completing a numerical continuous feature

# Completing an Age using correlation between Age, Gender and PClass
guess_ages = np.zeros((2, 3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) &
                               (dataset['Pclass'] == j + 1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i, j] = int(age_guess / 0.5 + 0.5) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[(dataset['Age'].isnull()) & (dataset['Sex'] == i) &
                        (dataset.Pclass == j + 1), 'Age'] = guess_ages[i, j]

# Creating AgeBand feature
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)

# Replacing Age with ordinals based on AgeBand
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 5

# Dropping AgeBands as unnecessary
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]

# Combining Parch and SibSp into one feature FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1


# # Creating IsAlone feature from FamilySize
# for dataset in combine:
#     dataset['IsAlone'] = 0
#     dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Another approach to family feature
def Fam_label(s):
    if (s >= 2) & (s <= 4):
        return 2
    elif ((s > 4) & (s <= 7)) | (s == 1):
        return 1
    elif s > 7:
        return 0


for dataset in combine:
    dataset['FamilyLabel'] = dataset['FamilySize'].apply(Fam_label)

# Dropping Parch, SibSp, and FamilySize features in favor of IsAlone (FamilyLabel)
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

# Creating  an artificial feature combining Pclass and Age
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

# Completing a categorical feature Embarked

# Finding mode at completed part of data
freq_port = train_df.Embarked.dropna().mode()[0]

# Filling in the null values with the mode
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)

# Completing and converting a numeric feature with mode
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# Creating FareBands feature
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# Replacing Fare with ordinals based on FareBand
for dataset in combine:
    dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
    dataset.loc[dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

# Dropping FareBand as unnecessary
train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]

X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Categorical features extraction
X_train_categ = X_train[['Embarked', 'Title', 'Deck']]
X_test_categ = X_test[['Embarked', 'Title', 'Deck']]

# Creation of binary features from categorical
enc = DictVectorizer()
X_train_categ = enc.fit_transform(X_train_categ.to_dict('records'))
X_test_categ = enc.transform(X_test_categ.to_dict('records'))

X_train = np.hstack([X_train.drop(['Embarked', 'Title', 'Deck'], axis=1), X_train_categ.toarray()])
X_test = np.hstack([X_test.drop(['Embarked', 'Title', 'Deck'], axis=1), X_test_categ.toarray()])

# # PCA
# pca = PCA()
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)


# Machine learning

start_time = datetime.datetime.now()
# Finding best metaparams for  using cross-validation
# param_grid = {'criterion': ['gini', 'entropy'],
#               # scoring methodology; two supported formulas for calculating information gain - default is gini
#               'n_estimators': [26],
#               # 'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12],
#               # 'splitter': ['best', 'random'], #splitting methodology; two supported strategies - default is best
#               'max_depth': [2, 4, 6, 8, 10, 3, 5],  # max depth tree can grow; default is none
#               'min_samples_split': [2, 5, 10, .03, .05],
#               # minimum subset size BEFORE new split (fraction is % of total); default is 2
#               'min_samples_leaf': [1, 5, 10, .03, .05],
#               # minimum subset size AFTER new split split (fraction is % of total); default is 1
#               'max_features': [None, 'auto', 'sqrt'],
#               # max features to consider when performing split; default none or all
#               'warm_start': [True, False],
#               'random_state': [0]
#               }
# tune_model = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, scoring='roc_auc',
#                           cv=KFold(n_splits=5, random_state=0))
#
# # Fitting classifier
# tune_model.fit(X_train, Y_train)
#
# # Getting prediction for test sample
# pred = tune_model.predict(X_test)

# Fitting classifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='entropy',
                             max_depth=6, max_features=None, max_leaf_nodes=None,
                             min_impurity_decrease=0.0, min_impurity_split=None,
                             min_samples_leaf=5, min_samples_split=0.03,
                             min_weight_fraction_leaf=0.0, n_estimators=10000,
                             n_jobs=None, oob_score=False, random_state=0, verbose=0,
                             warm_start=True).fit(X_train, Y_train)
# Getting prediction for test sample
pred = clf.predict(X_test)

# Writing submission to the file
submission = pd.DataFrame({
    'PassengerId': test_df['PassengerId'],
    'Survived': pred
})
submission.to_csv('tinkering_about.csv', index=False)
print(datetime.datetime.now() - start_time)

