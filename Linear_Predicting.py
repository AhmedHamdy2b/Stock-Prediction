import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

plt.style.use('bmh')

df = pd.read_csv('nflx.csv')
df.head(6)

plt.figure(figsize=(16, 8))
plt.title('Netflix')
plt.xlabel('Days')
plt.ylabel('Close Price USD($)')
plt.plot(df['Close'])

df = df[['Close']]
future_days = 25
df['Prediction'] = df[['Close']].shift(-future_days)
print(df.head(4))

x = np.array(df.drop(['Prediction'], 1))[:-future_days]
y = np.array(df['Prediction'])[:-future_days]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)
tree = DecisionTreeRegressor().fit(x_train, y_train)
lr = LinearRegression().fit(x_train, y_train)

x_future = df.drop(['Prediction'], 1)[:-future_days]
x_future = x_future.tail(future_days)
x_future = np.array(x_future)
print(x_future)

tree_prediction = tree.predict(x_future)

lr_prediction = lr.predict(x_future)

Predictions = lr_prediction

valid = df[x.shape[0]:]
valid['Predictions'] = Predictions
plt.figure(figsize=(16, 8))
plt.title('Model')
plt.xlabel('Days')
plt.ylabel('Close Price USD($)')
plt.plot(df['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Orig', 'Val', 'Pred'])
plt.show()
