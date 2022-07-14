
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, read_csv, Series
import scipy as scp
import scipy.stats as scpstats
from scipy.ndimage import shift
from ta import momentum
from sklearn.linear_model import LinearRegression

from sapi import Sapi

roll_period = 200
train_test_split_proportion = 0.8

raw_data_frame = read_csv('saved_data/data_saved0.csv', sep=",", index_col=False, header=2).iloc[:-1, :]
ask_data, bid_data = Series(raw_data_frame['ask'].values), Series(raw_data_frame['bid'].values)

rolling = DataFrame(ask_data).rolling(roll_period)

fig0, ax0 = plt.subplots()
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
# fig3, ax3 = plt.subplots()
# fig4, ax4 = plt.subplots()

rolling_diff = ask_data.diff(roll_period)
rolling_mean = rolling.mean().shift(1)
rolling_std = rolling.std().shift(1)

RSI = momentum.RSIIndicator(Series(ask_data), fillna=False)
rsi = RSI.rsi()

regression = LinearRegression()
train_data_length = int(train_test_split_proportion * len(ask_data))

regression.fit(DataFrame(ask_data[roll_period : round(train_data_length)]),
               rolling_diff[roll_period:round(rolling_diff.size * train_test_split_proportion)])

ax0.plot(ask_data[:train_data_length])
regression_prediction = regression.predict(DataFrame(ask_data[train_data_length:]))
ax0.plot(
    [x for x in range(train_data_length,
                      train_data_length + regression_prediction.size)],
    [ask_data[int(train_test_split_proportion * len(ask_data))] + regression_prediction[i] for i in range(len(regression_prediction))]
)
ax0.plot(
    [x for x in range(train_data_length,
                      train_data_length + regression_prediction.size)], ask_data[int(train_test_split_proportion * len(ask_data)):]
)
ax0.plot(rolling_mean)
ax0.plot([ask_data[i] if rsi[i] < 30 else 22000 for i in range(len(rsi))], 'go')
ax0.plot([ask_data[i] if rsi[i] > 70 else 22000 for i in range(len(rsi))], 'ro')
ax1.plot(rolling_std)
ax2.plot(rsi)
plt.show()

credentials = {}
with open('secrets.txt') as file:
    raw_data = file.readlines()
    credentials['email'] = raw_data[0].strip()
    credentials['password'] = raw_data[1].strip()
    credentials['totp'] = raw_data[2].strip()

btc_sapi = Sapi(credentials, 'BTC', 1)

balance = {'usd': 50, 'btc': 0}
btc_sapi.collect_data(10, 60, 'append')

while True:
    rsi = list(RSI.rsi())[-1]
    btc_sapi.collect_data(1, 20, 'append')
    RSI = momentum.RSIIndicator(Series(btc_sapi.ask_prices), fillna=False)
    print(rsi)

    balance = btc_sapi.get_balance('BTC')
    if rsi < 30 and balance['usd'] > 2:
        print(rsi)
        print('\u001b[32mBUY\u001b[0m')
        btc_sapi.post_transaction(balance['usd'] / 5, 'buy')
    elif rsi > 70 and balance['crypto']:
        print(rsi)
        print('\u001b[31mSELL\u001b[0m')
        btc_sapi.post_transaction(balance['crypto'] / 2, 'sell')