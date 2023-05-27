import numpy as np
import pandas as pd
import matplotlib as mp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('MARUTI.csv')

print(df)

print(df.head())

print(df.shape)
