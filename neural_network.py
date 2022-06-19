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
    trail_size = 50
    prediction_delay = 30
    batch_size = 32
    epochs = 10

    model = Sequential([
        Input(shape=(2, trail_size)),
        LSTM(300, activation='relu', return_sequences=True),
        LSTM(300, activation='relu', return_sequences=False),
        Dense(100, input_dim=300, activation='relu'),
        Dense(100, input_dim=100, activation='relu'),
        Dense(2, input_dim=100, activation='relu')
    ])

    def pre_model_transform(self, data, **kwargs):
        #return np.vectorize(math.log)(data)
        # avg_len = kwargs['avg_len']
        # if kwargs['arg_type'] == 'x':
        #     transformed_data = []
        #     for i in range(len(data) - avg_len):
        #         transformed_data.append((data[i:i+avg_len][0].mean(), data[i:i+avg_len][1].mean()))
        #     return np.array(transformed_data)
        # return data[:-avg_len]
        return data

    def post_model_transform(self, data, **kwargs):
        #return np.vectorize(math.exp)(data)
        return data

    def train(self):
        data = pandas.read_csv('saved_data/data_saved2.csv', sep=',', index_col=False, header=2)

        x = self.pre_model_transform(np.array([(
            data['ask'].iloc[i:i + self.trail_size],
            data['bid'].iloc[i:i + self.trail_size]
        ) for i in range(data.shape[0] - self.trail_size - self.prediction_delay)]), arg_type='x', avg_len=20)

        y = self.pre_model_transform(np.array([(
            data['ask'].iloc[i + self.prediction_delay],
            data['bid'].iloc[i + self.prediction_delay]
        ) for i in range(self.trail_size, data.shape[0] - self.prediction_delay)]), arg_type='y', avg_len=20)

        x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True, test_size=0.2)
        self.model.compile(loss='mean_squared_logarithmic_error', optimizer=Adagrad(0.005, decay=0.0002),
                           metrics=['accuracy', 'mean_absolute_error'])
        history = self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1).history

        transformed_x_test, transformed_y_test = self.post_model_transform(x_test), self.post_model_transform(y_test)
        fig, ax = plt.subplots()

        get_last_arg = lambda np_array: np_array[:, :, -1]
        ax.plot(get_last_arg(transformed_x_test), transformed_y_test, 'ro')
        ax.plot(get_last_arg(transformed_x_test), self.model.predict(transformed_x_test), 'bo')
        # errors = np.empty((0, 2), float)
        # for i, curr_x, curr_pred, curr_y in zip(range(len(x)), self.post_model_transform(x), self.post_model_transform(self.model.predict(x)), self.post_model_transform(y)):
        #     is_test_value = curr_x in transformed_x_test and curr_y in transformed_y_test
        #     styles = ['ro', 'bo']
        #     if is_test_value:
        #         errors = np.append(np.array(errors, ndmin=2), [curr_pred - curr_y], axis=0)
        #         print(f"Test {i + 1}:\n\t"
        #               f"X (last): {curr_x[:, -1]}\n\t"
        #               f"X (first): {curr_x[:, 0]}\n\t"
        #               f"Prediction: {curr_pred}\n\t"
        #               f"Actual value: {curr_y}\n\t"
        #               f"Error: {np.array(errors, ndmin=2)[-1, :] if errors.size != 0 else '-'}")
        #
        #         ax.plot(i, curr_x[:, -1][0], styles[0], markeredgewidth=0.001)
        #         # ax.plot(i, curr_x[:, -1][1], styles[0], markeredgewidth=0.001)
        #         ax.plot(i + self.prediction_delay, curr_pred[0], styles[1], markeredgewidth=0.001)
        #         # ax.plot(i + self.prediction_delay, curr_pred[1], styles[1], markeredgewidth=0.001)

        # abs_errors = np.absolute(np.array(errors))
        # print('dim 0', abs_errors[:, 0].mean(), abs_errors[:, 0].std())
        # print('dim 1', abs_errors[:, 1].mean(), abs_errors[:, 1].std())
        plt.show()
        print(history['mean_absolute_error'])

    def predict(self, data):
        transformed_data = self.pre_model_transform(data)
        returned_data = self.model.predict(transformed_data)
        return self.post_model_transform(returned_data)

if __name__ == '__main__':
    model = NeuralNet()
    model.train()
