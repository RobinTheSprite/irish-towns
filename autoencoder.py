import numpy as np
from tensorflow.keras.layers import Input, Conv1D, Dense, BatchNormalization
from tensorflow.keras.models import Sequential, Model
import pickle

CHARSET = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú']

def to_string(encoded_name):
    decoded = []
    for char in encoded_name:
        decoded.append(CHARSET[round(char * len(CHARSET))])

    return ''.join(decoded).strip()

encoding_dimension = 12
name_shape = (50,)

input_name = Input(name_shape)

model = Sequential()
model.add(input_name)
model.add(Dense(name_shape[0], activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(name_shape[0], activation='relu'))

autoencoder = Model(input_name, model(input_name))

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

training_data = pickle.load(open("float.pickle", "rb"))
testing_data = pickle.load(open("float-testing.pickle", "rb"))

autoencoder.fit(training_data, training_data,
                epochs=200,
                batch_size=50,
                shuffle=True,
                validation_data=(testing_data, testing_data))

decoded_names = autoencoder.predict(testing_data)

for i in range(10):
    print(f"Original:\n{to_string(testing_data[i])}\nDecoded:\n{to_string(decoded_names[i])}")
    print()