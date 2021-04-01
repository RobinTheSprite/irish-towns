import random
import pickle
import numpy as np

CHARSET = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú']

def get_unique_names(f, filename):
    filtered = set()
    for line in f:
        filtered.add(line)

    filtered = sorted(filtered)
    f1 = open(filename, "w")
    f1.writelines(filtered)


def normalize_data(f, filename):
    encodings = []
    for line in f:
        line = line.strip("\n")
        line = line.ljust(50)

        numeric = []
        for char in line:
            numeric.append(str(CHARSET.index(char) / len(CHARSET)))

        out = " ".join(numeric)
        out += "\n"
        encodings.append(out)

    f1 = open(filename, "w")
    f1.writelines(encodings)


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
    normalize_data(f, "random-strings-encoded.txt")

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

def pickle_data(f, not_irish_file):
    irish_data = []
    for line in f:
        line = line.strip("\n")
        name = line.split(" ")
        name = list(map(float, name))
        irish_data.append(name)

    f.close()

    irish_labels = list(1 for _ in range(len(irish_data)))

    irish_data = list(zip(irish_data, irish_labels))

    not_irish_data = []
    for line in not_irish_file:
        line = line.strip("\n")
        name = line.split(" ")
        name = list(map(float, name))
        not_irish_data.append(name)

    not_irish_file.close()

    not_irish_labels = list(0 for _ in range(len(not_irish_data)))

    not_irish_data = list(zip(not_irish_data, not_irish_labels))

    combined = irish_data + not_irish_data

    random.shuffle(combined)

    data = []
    labels = []
    for name, label in combined:
        data.append(name)
        labels.append(label)

    data = np.array(data).reshape((-1, 50, 1))
    labels = np.array(labels)

    f = open("data.pickle", "wb")
    pickle.dump(data, f)
    f.close()
    f = open("labels.pickle", "wb")
    pickle.dump(labels, f)
    f.close()


# normalize_data(open("irish-towns-training.txt", "r"), "irish-towns-encoded.txt")
# normalize_data(open("english-towns-training.txt", "r"), "english-towns-encoded.txt")

pickle_data(open("irish-towns-encoded.txt", "r"), open("english-towns-encoded.txt", "r"))