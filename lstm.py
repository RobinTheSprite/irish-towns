# lstm autoencoder recreate sequence
import numpy as np
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import OneHotEncoder
import pickle

CHARSET = [' ', '#', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú']

onehot_labels = OneHotEncoder(sparse=False)
onehot_labels = onehot_labels.fit_transform(list([i] for i in range(len(CHARSET))))

data = pickle.load(open("irish-towns-ngram-training-data.pickle", "rb"))
labels = pickle.load(open("irish-towns-ngram-training-labels.pickle", "rb"))

# reshape input into [samples, timesteps, features]
sequence_length = data.shape[1]
data = data.reshape((data.shape[0], sequence_length, 1))

def define_model():
    model = Sequential()
    model.add(LSTM(100, input_shape=(sequence_length,1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(100))
    model.add(Dropout(0.2))
    model.add(Dense(len(CHARSET), activation='softmax'))

    return model


def train():
    model = define_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    # fit model
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


def generate():
    model = define_model()
    model.load_weights("lstm-weights.hdf5")
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    name = "lis"
    seed = name
    for _ in range(50):
        x = np.array(list(CHARSET.index(char) / len(CHARSET) for char in seed))
        x = x.reshape((1, len(x), 1))

        prediction = model.predict(x)
        result = CHARSET[sample(prediction, 0.5)]
        name += result

        seed += result
        seed = seed[1:]

    print(name)

# train()
generate()