# data analysis and wrangling
import pandas as pd
import numpy as np

# time measurement
import datetime

# chdir
import os

os.chdir('/home/mikhail/PycharmProjects/kaggle/HousePrices')

data_train = pd.read_csv('input/train.csv')
data_test = pd.read_csv('input/test.csv')
data = [data_train, data_test]
