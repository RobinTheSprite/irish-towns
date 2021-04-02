import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import \
    BatchNormalization, Input, Dense, \
    Activation, Flatten, Conv1D, MaxPooling1D, \
    Reshape
import pickle


name_shape = (50, 1)
noise_shape = (100,)

discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

generator = build_generator()
name = generator(Input(shape = noise_shape))

discriminator.trainable = False

valid = discriminator(name)

combined = Model(Input(shape = noise_shape), valid)
discriminator.compile(loss="binary_crossentropy", optimizer="adam")


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
    model.add(BatchNormalization(momentum=0.8))
    model.add(MaxPooling1D(strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv1D(50, 2, activation="relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dense(50))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1, activation="sigmoid"))

    name = Input(shape=name_shape)
    validity = model(name)

    return Model(name, validity)


def train(epochs, batch_size=32, save_interval=50):
    data = pickle.load(open("data.pickle", "rb"))

    valid_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        i = np.random.randint(0, data.shape[0], batch_size)
        real_names = data[i]

        noise = np.random.normal(0.5, 0.5, (batch_size, noise_shape))
        fake_names = generator.predict(noise)

        discriminator.train_on_batch(real_names, valid_labels)
        discriminator.train_on_batch(fake_names, fake_labels)
        # loss_combined = 0.5 * np.add(loss_real, loss_fake)

        combined.train_on_batch(noise, valid_labels)

        if epoch % save_interval == 0:
            save()


def save():
    pass

# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
# model.fit(data, labels, batch_size=50, validation_split=0.1, epochs=200)