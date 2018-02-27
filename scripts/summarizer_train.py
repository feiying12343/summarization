import os
import pickle

from keras import Sequential
from keras.layers import Dropout, Lambda, Activation
from keras.layers import LSTM
from keras.optimizers import SGD


def train_model(trainX, trainY):
    model = Sequential()
    model.add(LSTM(input_dim, input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(input_dim, input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(output_dim, input_shape=(timesteps, input_dim), return_sequences=True))
    model.add(Lambda(lambda x: x[:, -output_len:, :]))
    model.add(Activation("softmax"))
    sgd = SGD(lr=2)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    print(model.summary())
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    model.save(path + "models/" + fname + ".model")


if __name__ == '__main__':
    timesteps = 239
    output_dim = 3
    output_len = 120
    fname = "gossip"
    if "nt" == os.name:
        path = "E:/Martin/PyCharm Projects/Summarization/"
    else:
        path = "/home/matulma4/summarization/"
    X, y = pickle.load(open(path + "model_output/samples/" + fname + "_samples.pickle", "rb"))
    input_dim = X.shape[1]
    train_model(X, y)
