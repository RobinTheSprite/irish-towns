from sklearn.preprocessing import OneHotEncoder

CHARSET = [' ', '#', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'é', 'í', 'ó', 'ú']

onehot_labels = OneHotEncoder(sparse=False)
onehot_labels = onehot_labels.fit_transform(list([i] for i in range(len(CHARSET))))