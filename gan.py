import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import \
    BatchNormalization, Input, Dense, \
    Activation, Flatten, Conv1D, MaxPooling1D, \
    Reshape
import pickle


name_shape = (50, 1)
noise_shape = (100,)


def build_generator():
    model = Sequential()

    model.add(Dense(256, input_shape = noise_shape, activation="relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(256))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(np.product(name_shape), activation="tanh"))
    model.add(Reshape(name_shape))

    noise = Input(shape=noise_shape)
    name = model(noise)

    return Model(noise, name)


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