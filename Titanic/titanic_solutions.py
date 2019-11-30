# data analysis and wrangling
import pandas as pd
import numpy as np
# import random as rnd

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
from catboost import CatBoostClassifier
# chdir
import os

os.chdir('/home/mikhail/PycharmProjects/kaggle/Titanic/data')

# Data analysis

# Reading data sets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
combine = [train_df, test_df]

# Getting a general look at features
print(train_df.info())
print('_' * 40)
print(test_df.info())

# Getting summary about features
print(train_df.describe())
print(train_df.describe(include=['O']))

# Proving correlation purposes
print(train_df[['Pclass', 'Survived']].groupby(
    ['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_' * 40)
print(train_df[["Sex", "Survived"]].groupby(
    ['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_' * 40)
print(train_df[["SibSp", "Survived"]].groupby(
    ['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False))
print('_' * 40)
print(train_df[["Parch", "Survived"]].groupby(
    ['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# Visualizing data

# Numerical
grid = sns.FacetGrid(train_df, col='Survived')
grid.map(plt.hist, 'Age', bins=20)
plt.show()

# Numerical and ordinal
grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

# Categorical
grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
plt.show()

# Categorical and numerical
grid = sns.FacetGrid(train_df, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()
plt.show()


# Wrangling data

# Dropping features
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Creating new feature extracting from existing
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

# Analysing correlation between Title and Sex
print(pd.crosstab(train_df['Title'], train_df['Sex']))

# Uniting rare titles and replacing some titles with a more common name
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
         'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

# Analysing correlation between Title and Survived
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

# Converting categorical titles to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

# Dropping features
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

# Converting a categorical feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Completing a numerical continuous feature

# Visualization of Age, Gender and PClass correlation
grid = sns.FacetGrid(train_df, row='Pclass', col='Sex', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()
plt.show()

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

# Correlation between AgeBand and Survived
print(train_df[['AgeBand', 'Survived']].groupby(
    ['AgeBand'], as_index=False).mean().sort_values(
    by='AgeBand', ascending=True))

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

# Combining Parch and Fare into one feature FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Correlation between FamilySize and Survived
print(train_df[['FamilySize', 'Survived']].groupby(
    ['FamilySize'], as_index=False).mean().sort_values(
    by='Survived', ascending=False))

# Creating IsAlone feature from FamilySize
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Correlation between IsAlone and Survived
print(train_df[['IsAlone', 'Survived']].groupby(
    ['IsAlone'], as_index=False).mean())

# Dropping Parch, SibSp, and FamilySize features in favor of IsAlone
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

# Correlation between Embarked and Survived
print(train_df[['Embarked', 'Survived']].groupby(
    ['Embarked'], as_index=False).mean().sort_values(
    by='Survived', ascending=False))

# Converting categorical feature Embarked to numeric
for dataset in combine:
    dataset['Embarked'] = \
        dataset['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

# Completing and converting a numeric feature with mode
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

# Creating FareBands feature
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)

# Correlation between FareBand and Survived
print(train_df[['FareBand', 'Survived']].groupby(
    ['FareBand'], as_index=False).mean().sort_values(
    by='FareBand', ascending=True))

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


# Machine learning

# Train and test data
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
Y_pred_log = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

# Coefficient extraction from logistic regression
coeff_df = pd.DataFrame(train_df.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])

# Correlation of features with Survived based on logistic regression
coeff_df.sort_values(by='Correlation', ascending=False)

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_pred_svc = svc.predict(X_test)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)

# K Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred_knn = knn.predict(X_test)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
Y_pred_gaussian = gaussian.predict(X_test)
acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

# Perceptron
perceptron = Perceptron()
perceptron.fit(X_train, Y_train)
Y_pred_perceptron = perceptron.predict(X_test)
acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred_linear_svc = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(X_train, Y_train)
Y_pred_sgd = sgd.predict(X_test)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
Y_pred_decision_tree = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, Y_train)
Y_pred_random_forest = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

# Gradient Boosting
grad_boost = GradientBoostingClassifier()
grad_boost.fit(X_train, Y_train)
Y_pred_grad_boost = grad_boost.predict(X_test)
acc_grad_boost = round(grad_boost.score(X_train, Y_train) * 100, 2)

# MLP
mlp = MLPClassifier()
mlp.fit(X_train, Y_train)
Y_pred_mlp = mlp.predict(X_test)
acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)

# XGB
xgb = XGBClassifier()
xgb.fit(X_train, Y_train)
Y_pred_xgb = xgb.predict(X_test)
acc_xgb = round(xgb.score(X_train, Y_train) * 100, 2)

# Model evaluation
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression',
              'Random Forest', 'Naive Bayes', 'Perceptron',
              'Stochastic Gradient Decent', 'Linear SVC',
              'Decision Tree', 'Gradient Boosting', 'MLP', 'XGB'],
    'Score': [acc_svc, acc_knn, acc_log,
              acc_random_forest, acc_gaussian, acc_perceptron,
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_grad_boost,
              acc_mlp, acc_xgb]})
print(models.sort_values(by='Score', ascending=False))

# # Writing submission to the file
# submission = pd.DataFrame({
#         'PassengerId': test_df['PassengerId'],
#         'Survived': Y_pred_random_forest
#     })
# submission.to_csv('prediction_solution.csv', index=False)
