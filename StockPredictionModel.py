import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

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


features = ['Open', 'Low', 'High', 'Close', 'Volume']
 
for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.distplot(df[col])


for i, col in enumerate(features):
  plt.subplot(2,3,i+1)
  sb.boxplot(df[col])



splitted = df['Date'].str.split('-', expand=True)

df['day'] = splitted[1].astype('int')
df['month'] = splitted[0].astype('int')
df['year'] = splitted[2].astype('int')
 
print(df.head())


df['quarter'] = np.where(df['month']%3==0,1,0)
print(df)


df['open-close']  = df['Open'] - df['Close']
df['low-high']  = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)


features = df[['open-close', 'low-high', 'quarter']]
target = df['target']
 
scaler = StandardScaler()
features = scaler.fit_transform(features)
 
X_train, X_test, Y_train, Y_test = train_test_split(
    features, target, test_size=0.2, random_state=2022)
print(X_train.shape, X_test.shape)


models = [LogisticRegression(), SVC(kernel='poly', probability=True), XGBClassifier()]
print('\n')
for i in range(3):
 models[i].fit(X_train, Y_train)
 models[i].fit(X_train, Y_train) 
 print(f'{models[i]} : ')
 print('Accuracy(Training..) : ', metrics.roc_auc_score(Y_train, models[i].predict_proba(X_train)[:,1]))
 print('Accuracy(Testing..) : ', metrics.roc_auc_score(Y_test, models[i].predict_proba(X_test)[:,1]))
 print('\n')

