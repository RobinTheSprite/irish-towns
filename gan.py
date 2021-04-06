import os
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import \
    BatchNormalization, Input, Dense, \
    Activation, Flatten, Conv1D, MaxPooling1D, \
    Reshape, LeakyReLU, Dropout
import pickle
from sklearn.preprocessing import OneHotEncoder

CHARSET = (' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú')

onehot_labels = OneHotEncoder(sparse=False)
onehot_labels = onehot_labels.fit_transform(list([i] for i in range(len(CHARSET))))

name_shape = (50, 32)
noise_shape = (100,)

DATA = pickle.load(open("data.pickle", "rb"))

if os.path.exists("results.txt"):
  os.remove("results.txt")


def build_generator():
    model = Sequential()

    model.add(Dense(128, input_shape = noise_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(1024))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU())
    model.add(Dropout(0.4))
    model.add(Dense(np.product(name_shape), activation="tanh"))
    model.add(Reshape(name_shape))

    noise = Input(shape=noise_shape)
    name = model(noise)

    model.summary()

    return Model(noise, name)


def build_discriminator():
    model = Sequential()
    model.add(Conv1D(50, 2, input_shape = name_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Conv1D(100, 2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(0.2))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))

    name = Input(shape=name_shape)
    validity = model(name)

    model.summary()

    return Model(name, validity)


def train(epochs, batch_size=32, save_interval=50):
    data = DATA
    data = data * 2 - 1

    valid_labels = np.ones((batch_size, 1))
    fake_labels = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        i = np.random.randint(0, data.shape[0], batch_size)
        real_names = data[i]

        noise = np.random.normal(0, 1, (batch_size, noise_shape[0]))
        fake_names = generator.predict(noise)

        loss_real = discriminator.train_on_batch(real_names, valid_labels)
        loss_fake =  discriminator.train_on_batch(fake_names, fake_labels)
        loss_combined = 0.5 * np.add(loss_real, loss_fake)

        generator_loss = combined.train_on_batch(noise, valid_labels)

        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, loss_combined[0], 100*loss_combined[1], generator_loss))

        if epoch % save_interval == 0:
            save(epoch)


def save(epoch):
    num_of_names = 10
    noise = np.random.normal(0, 1, (num_of_names, noise_shape[0]))
    names_encoded = generator.predict(noise)

    names = []
    for name in names_encoded.tolist():
        name_str = str()
        for char_probabilities in name:
            name_str += CHARSET[np.argmax(char_probabilities)]


        name_str.strip()
        name_str += "\n"
        names.append(name_str)

    f = open("results.txt", "a")

    f.write(f"Epoch {epoch}:\n")
    f.write("\n")
    f.writelines(names)
    f.write("\n\n")
    f.close()


discriminator = build_discriminator()
discriminator.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

generator = build_generator()
z = Input(shape = noise_shape)
name = generator(z)

discriminator.trainable = False

prediction = discriminator(name)

combined = Model(z, prediction)
combined.compile(loss="binary_crossentropy", optimizer="adam")

train(epochs = 1000, save_interval = 100)

# generator.save("generator.model.h5")