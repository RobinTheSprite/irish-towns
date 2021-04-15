# lstm autoencoder recreate sequence
import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder
import pickle
from charset import CHARSET, onehot_labels

data = pickle.load(open("irish-towns-ngram-training-data.pickle", "rb"))
labels = pickle.load(open("irish-towns-ngram-training-labels.pickle", "rb"))

sequence_length = data.shape[1]
# [samples, timesteps, features]
data = data.reshape((data.shape[0], sequence_length, 1))

def define_model():
    model = Sequential()
    model.add(LSTM(128, input_shape=(sequence_length,1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(Dense(len(CHARSET), activation='softmax'))

    return model


def train():
    model = define_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(data, labels, epochs=100, batch_size=64)
    model.save_weights("lstm-weights.hdf5", overwrite=True)


# from https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/8.1-text-generation-with-lstm.ipynb
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    preds = preds.reshape((preds.shape[1],))
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def generate(number_of_names=1):
    model = define_model()
    model.load_weights("lstm-weights.hdf5")
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    for _ in range(number_of_names):
        name = ''.join('#' for _ in range(sequence_length))
        seed = name
        for _ in range(25):
            x = np.array(list(CHARSET.index(char) / len(CHARSET) for char in seed))
            x = x.reshape((1, len(x), 1))

            prediction = model.predict(x)
            result = CHARSET[sample(prediction, 0.5)]
            name += result

            seed += result
            seed = seed[1:]

        name = name[sequence_length:]
        print(name)

train()
generate(10)