from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('fivethirtyeight')

df = pd.read_csv('nflx.csv')

actual_price = df.tail(1)
df = df.head(len(df)-1)
days = []
adj_close_prices = []

df_days = df.loc[:, 'Date']
df_adj_close = df.loc[:, 'Adj Close']

for day in df_days:
    days.append([int(day.split('-')[2])])
for adj_close_price in df_adj_close:
    adj_close_prices.append(float(adj_close_price))

lin_svr = SVR(kernel='linear', C=1000.0)
lin_svr.fit(days, adj_close_prices)

poly_svr = SVR(kernel='poly', C=1000.0, degree=2)
poly_svr.fit(days, adj_close_prices)

rbf_svr = SVR(kernel='rbf', C=1000.0, gamma=0.85)
rbf_svr.fit(days, adj_close_prices)

plt.figure(figsize=(16, 8))
plt.scatter(days, adj_close_prices, color='black', label='Data')
plt.plot(days, rbf_svr.predict(days), color='green', label='RBF_MODEL')
plt.plot(days, poly_svr.predict(days), color='orange', label='Polynomial_MODEL')
plt.plot(days, lin_svr.predict(days), color='blue', label='Linear_MODEL')
plt.xlabel('Days')
plt.ylabel('Adj Close Price($)')
plt.legend()
plt.show()

day = [[2]]
print("the rbf svr predicted price", rbf_svr.predict(day))
print("the poly svr predicted price", poly_svr.predict(day))
print("the linear svr predicted price", lin_svr.predict(day))