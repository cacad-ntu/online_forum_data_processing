from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
import numpy as np


class RNN:

    def __init__(self, x_train, x_test, y_train, y_test, max_features, max_len=80, num_neurons=128):

        self.x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        self.x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        self.y_test = y_test
        self.y_train = y_train

        self.model = Sequential()
        self.model.add(Embedding(max_features, num_neurons))
        self.model.add(LSTM(num_neurons, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(len(y_test[0]), activation='softmax'))

    def start_train(self, batch_size, epochs):

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Start Training RNN')
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(self.x_test, self.y_test), shuffle=True)

        score, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)

    def predict_one_data(self, x_test):

        predict = self.model.predict(np.
                                     array(x_test))
        return np.argmax(predict, axis=1)
