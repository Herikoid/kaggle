# data analysis and wrangling
import pandas as pd
import numpy as np

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# time measurement
import datetime

# chdir
import os

os.chdir('/home/mikhail/PycharmProjects/kaggle/HousePrices')

# Data analysis

# reading data
data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')
data = [data_train, data_test]

# getting a general look at features
print(data_train.info())
print('_' * 40)
print(data_test.info())

# getting summary about features
print(data_train.describe())
print(data_train.describe(include=['O']))

# Visualization

#