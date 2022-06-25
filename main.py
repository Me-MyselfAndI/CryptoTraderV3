import logging
from time import sleep, time
from threading import Thread, Event
import numpy as np
import math

from neural_network import NeuralNet
from sapi import Sapi


def update_price_data(sapi, interval, trials_amount, save_interval=50, data_path='saved_data/data.csv'):
    while True:
        for i in range(math.ceil(trials_amount / save_interval)):
            sapi.collect_data(save_interval, interval, 'append')
            sapi.export_data(data_path, interval)


def train_model(model_creator, sapi, measurement_interval, data_path, delay):
    while True:
        try:
            model_creator.train(data_path)
            training_ended_time = time()
            while True:
                if time() <= training_ended_time + delay:
                    break
                print(
                    f'\u001b[32mPrediction {model_creator.prediction_delay * measurement_interval} seconds forward is {model_creator.predict(np.array([sapi.ask_prices, sapi.bid_prices]))}\u001b[0m')
                sleep(model_creator.prediction_delay * measurement_interval)
                print(f'\u001b[32mCurrent prices are: {sapi.ask_prices[-1], sapi.bid_prices[-1]}\u001b[0m\n\n')
        except Exception as err:
            print(err)


if __name__ == '__main__':
    interval = 15
    trials_amount = 1000

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
    # if mode == 0:
    #     for i in range(2000):
    #         print(f'{i}/2000')
    #         btc_sapi.collect_data(15, interval, 'append')
    #         btc_sapi.export_data('data_saved0.csv', interval)
    #
    #     model = NeuralNet()
    #     model.train()
    #     data = np.array([btc_sapi.ask_prices[-model.trail_size], btc_sapi.bid_prices[-model.trail_size]])
    # elif mode == 1:
    #     data = btc_sapi.read_file('data_saved0.csv').to_numpy()
    #     print(data)
    #     model = NeuralNet()
    #     model.train()

    mode = int(input('Enter mode'))
    if mode == 0:
        # Create a price updating thread
        btc_sapi.collect_data(trials_amount, interval, 'erase')
    elif mode == 1:
        data = btc_sapi.read_file('data_saved0.csv').to_numpy()
        btc_sapi.ask_prices, btc_sapi.bid_prices = data[0], data[1]

    # Reset the price updating thread to run constantly on append
    data_generation_thread = Thread(target=update_price_data, args=[btc_sapi, interval, trials_amount])
    data_generation_thread.start()

    # Run model-updating thread constantly
    model_creator = NeuralNet(200, 500)
    model_update_thread = Thread(target=train_model, args=[model_creator, btc_sapi, interval, 'saved_data/data.csv', 1200])
    model_update_thread.start()

    # # Run prediction-and-trading thread constantly
    # while True:
    #
