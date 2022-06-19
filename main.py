import logging
from time import sleep
from threading import Thread, Event
import numpy as np

from neural_network import NeuralNet
from sapi import Sapi

if __name__ == '__main__':
    interval = 10

    logging.basicConfig()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    credentials = {}
    with open('secrets.txt') as file:
        raw_data = file.readlines()
        credentials['email'] = raw_data[0].strip()
        credentials['password'] = raw_data[1].strip()
        credentials['totp'] = raw_data[2].strip()

    btc_sapi = Sapi(credentials, 'BTC', 5, logger=logger)
    # mode = int(input('Enter mode'))
    # if mode == 0:
    #     for i in range(2000):
    #         print(f'{i}/2000')
    #         btc_sapi.collect_data(15, interval, 'append')
    #         btc_sapi.export_data('data.csv', interval)
    #
    #     model = NeuralNet()
    #     model.train()
    #     data = np.array([btc_sapi.ask_prices[-model.trail_size], btc_sapi.bid_prices[-model.trail_size]])
    # elif mode == 1:
    #     data = btc_sapi.read_file('data.csv').to_numpy()
    #     print(data)
    #     model = NeuralNet()
    #     model.train()

    # Create a price updating thread
    # Run it N times, if needed, to generate initial data
    # Reset the price updating thread to run constantly on append
    # Run model-updating thread constantly
    # Run prediction-and-trading thread constantly