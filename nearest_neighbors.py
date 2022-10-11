import numpy as np


def KNN_test(X_train: np.ndarray, Y_train: np.ndarray, X_test: np.ndarray, Y_test: np.ndarray, K: int) -> float:
    assert X_test.shape[0] == Y_test.shape[0]
    assert X_train.shape[0] == Y_train.shape[0]

    correct_count = 0
    for i in range(X_test.shape[0]):
        distances = []
        for j in range(X_train.shape[0]):
            distances += [(np.linalg.norm(X_test[i] - X_train[j]), j)]

        distances.sort(key=lambda x: x[0])
        assert len(distances) >= K

        labels = Y_train[list(sample[1] for sample in distances[0:K])]
        values, counts = np.unique(labels, return_counts=True)
        predicted = values[np.argmax(counts)]

        if predicted == Y_test[i]:
            correct_count += 1

    return correct_count / X_test.shape[0]


def load_data(file_data):
    data = np.genfromtxt(file_data, skip_header=1, delimiter=',')
    X = []
    Y = []
    for row in data:
        temp = [float(x) for x in row]
        temp.pop(-1)
        X.append(temp)
        Y.append(int(row[-1]))
    X = np.array(X)
    Y = np.array(Y)
    return X,Y


if __name__ == "__main__":
    X,Y = load_data("nearest_neighbors_1.csv")
    acc = KNN_test(X,Y,X,Y,3)
    print("KNN:", acc)