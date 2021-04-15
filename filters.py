import random
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

CHARSET = [' ', '#', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú']

onehot_labels = OneHotEncoder(sparse=False)
onehot_labels = onehot_labels.fit_transform(list([i] for i in range(len(CHARSET))))


def get_unique_names(f, filename):
    filtered = set()
    for line in f:
        filtered.add(line)

    filtered = sorted(filtered)
    f1 = open(filename, "w")
    f1.writelines(filtered)


def onehot_encoding(char):
    return onehot_labels[CHARSET.index(char)]


def float_encoding(char):
    return CHARSET.index(char) / len(CHARSET)


def encode_data(f, filename, encoder):
    encodings = []
    for line in f:
        line = line.strip("\n")
        line = line.ljust(50)

        numeric = []
        for char in line:
            numeric.append(encoder(char))

        encodings.append(numeric)

    random.shuffle(encodings)
    encodings = np.array(encodings)

    f1 = open(filename, "wb")
    pickle.dump(encodings, f1)


def lowercase(f, filename):
    lowercased = []
    for line in f:
        lowercased.append(line.lower())

    f1 = open(filename, "w")
    f1.writelines(lowercased)


def make_ngram_sequences(f, file_pattern, sequence_length):
    data = []
    labels = []
    town_index = 0
    for town in f:
        town = town.strip()
        town = ''.join('#' for _ in range(sequence_length)) + town
        if town_index % 3 == 0:
            town = town.ljust(25)
        for i in range(len(town) - sequence_length):
            seq_in = town[i:i + sequence_length]
            seq_out = town[i + sequence_length]
            data.append(list(CHARSET.index(char) for char in seq_in))
            labels.append(onehot_labels[CHARSET.index(seq_out)])

        town_index += 1

    data = np.array(data)
    data = data / len(CHARSET)
    labels = np.array(labels)

    f.close()
    pickle.dump(data, open(f"{file_pattern}-data.pickle", "wb"))
    pickle.dump(labels, open(f"{file_pattern}-labels.pickle", "wb"))


make_ngram_sequences(open("irish-towns-training.txt", "r"), "irish-towns-ngram-training", 4)

