import random
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

CHARSET = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú']

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

def make_random_strings():

    strings = []
    for _ in range(1325):
        string_length = random.randint(2, 10)
        string = str()
        for _ in range(string_length):
            string += random.choice(CHARSET)

        string = string.strip()
        string += "\n"
        if not string.isspace():
            strings.append(string)

    f = open("random-strings.txt", "w")
    f.writelines(strings)
    f.close()
    f = open("random-strings.txt", "r")
    encode_data(f, "random-strings-encoded.txt", float_encoding)

def make_testing_data(input_file, training_data_file, testing_data_file):
    f = open(input_file, "r")
    names = list(f)
    names = list(map(str.strip, names))

    testing_names = []
    while len(names) > 1000:
        name = random.choice(names)
        names.remove(name)
        name += "\n"
        testing_names.append(name)

    f.close()

    f = open(training_data_file, "w")
    names = "\n".join(names)
    f.writelines(names)
    f.close()

    f = open(testing_data_file, "w")
    f.writelines(testing_names)


def make_ngram_sequences(f, sequence_length):
    data = []
    labels = []
    for town in f:
        town = town.strip()
        for i in range(len(town) - sequence_length):
            seq_in = town[i:i + sequence_length]
            seq_out = town[i + sequence_length]
            data.append(list(CHARSET.index(char) for char in seq_in))
            labels.append(onehot_labels[CHARSET.index(seq_out)])

    data = np.array(data)
    data = data / len(CHARSET)
    labels = np.array(labels)

    f.close()
    pickle.dump(data, open("irish-towns-ngram-training-data.pickle", "wb"))
    pickle.dump(labels, open("irish-towns-ngram-training-labels.pickle", "wb"))


make_ngram_sequences(open("irish-towns-training.txt", "r"), 3)
# encode_data(open("irish-towns-training.txt", "r"), "float.pickle", float_encoding)
