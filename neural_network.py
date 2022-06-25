import math

import numpy
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adagrad, Adam, Nadam, RMSprop
import pandas
from sklearn.model_selection import train_test_split


class NeuralNet:
    model = None

    def __init__(self, trail_size, prediction_delay, batch_size=32, epochs=10):
        self.trail_size = trail_size
        self.prediction_delay = prediction_delay
        self.batch_size = batch_size
        self.epochs = epochs

    def pre_model_transform(self, data, **kwargs):
        # return np.vectorize(math.log)(data)

        # avg_len = kwargs['avg_len']
        # if kwargs['arg_type'] == 'x':
        #     transformed_data = []
        #     for i in range(len(data) - avg_len):
        #         transformed_data.append((data[i:i+avg_len][0].mean(), data[i:i+avg_len][1].mean()))
        #     return np.array(transformed_data)
        # return data[:-avg_len]

        return data

    def post_model_transform(self, data, **kwargs):
        # return np.vectorize(math.exp)(data)

        return data

    def train(self, data_path, max_tolerable_err=400, min_runs=5):
        data = pandas.read_csv(data_path, sep=',', index_col=False, header=2)

        x = self.pre_model_transform(np.array([(
            data['ask'].iloc[i:i + self.trail_size],
            data['bid'].iloc[i:i + self.trail_size]
        ) for i in range(data.shape[0] - self.trail_size - self.prediction_delay)]))

        y = self.pre_model_transform(np.array([(
            data['ask'].iloc[i + self.prediction_delay],
            data['bid'].iloc[i + self.prediction_delay]
        ) for i in range(self.trail_size, data.shape[0] - self.prediction_delay)]))

        run = 0
        best_model = None
        best_performance = math.inf
        while run < min_runs:
            x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)
            model = Sequential([
                Input(shape=(2, self.trail_size)),
                LSTM(300, activation='relu', return_sequences=True),
                LSTM(300, activation='relu', return_sequences=False),
                Dense(100, input_dim=300, activation='relu'),
                Dense(100, input_dim=100, activation='relu'),
                Dense(2, input_dim=100, activation='relu')
            ])
            model.compile(loss='mean_squared_logarithmic_error', optimizer=Adagrad(0.005, decay=0.0002),
                          metrics=['accuracy', 'mean_absolute_error'])
            history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1).history

            transformed_x_test, transformed_y_test = self.pre_model_transform(x_test), self.pre_model_transform(
                y_test)

            mean_absolute_error = history['mean_absolute_error'][-1]
            if mean_absolute_error <= max_tolerable_err:
                run += 1
                if mean_absolute_error < best_performance:
                    best_performance = mean_absolute_error
                    best_model = model

        fig, ax = plt.subplots()
        transformed_x, transformed_y = self.pre_model_transform(x), self.pre_model_transform(y)
        get_last_arg = lambda np_array: np_array[:, :, -1]
        ax.plot(get_last_arg(transformed_x), transformed_y, 'ro')
        ax.plot(get_last_arg(transformed_x), best_model.predict(transformed_x), 'bo')
        plt.show()
        self.model = best_model

    def predict(self, data):
        transformed_data = self.pre_model_transform(
            np.array([data[:, i:i + self.trail_size] for i in range(data.shape[1] - self.trail_size)])
        )
        returned_data = self.model.predict(transformed_data)
        return self.post_model_transform(returned_data)[:, :, -1]


if __name__ == '__main__':
    model_handler = NeuralNet()
    model_handler.train('saved_data/data_saved2.csv', 250, 10)
