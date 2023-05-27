import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df = pd.DataFrame()

df = pd.read_csv('MARUTI.csv')

print(df.head())

print(df.shape)

print(df.describe())

print(df.info())

print(df.isna().sum())

df= df.fillna(0)

print(df.isna().sum())

plt.plot(df['Close'])
plt.ylabel('ClosePrice')
plt.show()

features = ['Open', 'Low', 'High', 'Close', 'Volume']
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])
plt.show()

for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])
plt.show()


splitted = df['Date'].str.split('-', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
 
print(df.head())