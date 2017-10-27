from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM


class RNN:

    def __init__(self, x_train, x_test, y_train, y_test, max_features, max_len=80):

        self.x_train = sequence.pad_sequences(x_train, maxlen=max_len)
        self.x_test = sequence.pad_sequences(x_test, maxlen=max_len)

        self.y_test = y_test
        self.y_train = y_train

        self.model = Sequential()
        self.model.add(Embedding(max_features, 128))
        self.model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
        self.model.add(Dense(len(y_test[0]), activation='softmax'))

    def start_train(self, batch_size, epochs):

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        print('Train...')
        self.model.fit(self.x_train, self.y_train, batch_size=batch_size, epochs=epochs,
                       validation_data=(self.x_test, self.y_test))

        score, acc = self.model.evaluate(self.x_test, self.y_test, batch_size=batch_size)
        print('Test score:', score)
        print('Test accuracy:', acc)
