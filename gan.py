import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten, Conv1D, MaxPooling1D
import pickle


name_shape = (50, 1)


def build_generator():
    pass


def build_discriminator():
    model = Sequential()
    model.add(Conv1D(50, 2, input_shape = name_shape, activation="relu"))
    model.add(MaxPooling1D(strides=2))
    model.add(Conv1D(50, 2, activation="relu"))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(Dense(1, activation="sigmoid"))

    name = Input(shape=name_shape)
    validity = model(name)

    return Model(name, validity)


def train(epochs, batch_size=32, save_interval=50):
    data = pickle.load(open("data.pickle", "rb"))
    labels = pickle.load(open("labels.pickle", "rb"))


def save():
    pass

# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(data, labels, batch_size=50, validation_split=0.1, epochs=200)