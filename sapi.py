import datetime
import logging
from time import sleep
import numpy as np
import pandas
import robin_stocks.robinhood as rs
import pyotp, math

class Sapi:
    ask_prices, bid_prices = np.array([]), np.array([])
    start_time, end_time = None, None
    logged_in = False

    def __init__(self, credentials, currency, precision, logger=None):
        totp = pyotp.TOTP(credentials['totp']).now()
        rs.login(credentials['email'], credentials['password'], mfa_code=totp)
        self.logged_in = True
        self.currency = currency
        self.precision = precision
        if logger is None:
            self.logger = logging
        else:
            self.logger = logger

    def read_file(self, filename):
        data = pandas.read_csv(filename, sep=',', index_col=False, header=1)
        return data

    def collect_row(self, interval):
        crypto_quote = rs.crypto.get_crypto_quote(self.currency)
        ask_price, bid_price = round(float(crypto_quote['ask_price']), self.precision), \
                               round(float(crypto_quote['bid_price']), self.precision)
        sleep(interval)
        return np.array([ask_price, bid_price])

    '''
    duration: number of intervals
    interval: number of seconds to wait between two requests to the server
    mode: 'erase' to clear old values and start over; 'append' to keep adding values; 'replace' to keep the current length and add new values, while deleting old, one by one
    '''
    def collect_data(self, duration, interval, mode='replace'):
        if not self.logged_in:
            logging.warning('User is not logged in')
            raise Exception('Authentication error')
        if mode == 'erase':
            self.ask_prices, self.bid_prices = np.array([]), np.array([])
            self.start_time = datetime.datetime.utcnow()

            def update_prices(ask_price, bid_price):
                self.ask_prices = np.append(self.ask_prices, ask_price)
                self.bid_prices = np.append(self.bid_prices, bid_price)
        elif mode == 'append':
            def update_prices(ask_price, bid_price):
                self.ask_prices = np.append(self.ask_prices, ask_price)
                self.bid_prices = np.append(self.bid_prices, bid_price)
        elif mode == 'replace':
            self.start_time = datetime.datetime.utcnow()
            if self.ask_prices.size < interval or self.bid_prices.size < interval:
                logging.warning('Cannot replace ask or bid prices, since they are shorter than the interval')
                raise Exception('Ask or bid prices\' length is smaller than interval')

            def update_prices(ask_price, bid_price):
                self.ask_prices = np.append(self.ask_prices[1:], round(float(ask_price), self.precision))
                self.bid_prices = np.append(self.bid_prices[1:], round(float(bid_price), self.precision))
        else:
            logging.warning(
                f'SAPI collect_data - wrong mode value; must be "erase", "append" or "replace", but {mode} was provided')
            raise Exception(f'Wrong mode value; must be "erase", "append" or "replace", but {mode} was provided')

        if self.start_time is None:
            self.start_time = datetime.datetime.utcnow()

        for time in range(duration):
            ask_price, bid_price = self.collect_row(interval)
            update_prices(ask_price, bid_price)

            print(f'{time+1}/{duration} price update\n\t'
                            f'New ask price: {ask_price}\n\t'
                            f'New bid price: {bid_price}\n\t'
                            f'Mode: "{mode.capitalize()}"\n\t')

        self.end_time = datetime.datetime.utcnow()

    def export_data(self, filename, interval):
        with open(filename, 'w') as file:
            file.write(f'Start Time: {self.start_time}; End Time: {self.end_time};\nSize: {self.ask_prices.size}; Interval: {interval}\n')
            file.write('ask,bid\n')
            data_string = ''
            for ask_price, bid_price in zip(self.ask_prices, self.bid_prices):
                data_string += str(ask_price) + ',' + str(bid_price) + '\n'
            file.write(data_string[:-1])

    def post_transaction(self, usd_quantity, mode):
        if mode == 'buy':
            response = rs.order_buy_crypto_by_price(symbol=self.currency, amountInDollars=round(usd_quantity, 3))
        elif mode == 'sell':
            response = rs.order_sell_crypto_by_price(symbol=self.currency, amountInDollars=round(usd_quantity, 3))
        else:
            raise Exception('Wrong mode')

        print(response)

