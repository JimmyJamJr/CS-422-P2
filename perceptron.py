import numpy as np


def perceptron_train(X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    n_samples, n_features = X.shape
    W = np.zeros(n_features)
    b = 0

    updated = True
    while updated:
        updated = False
        for i in range(n_samples):
            a = np.dot(X[i, :], W) + b
            if a * Y[i] <= 0:
                updated = True
                W = W + X[i, :] * Y[i]
                b = b + Y[i]

    return W, b


def perceptron_test(X_test: np.ndarray, Y_test: np.ndarray, w: np.ndarray, b: float) -> float:
    correct_count = 0
    for i in range(X_test.shape[0]):
        a = np.dot(X[i, :], w) + b
        if a * Y_test[i] > 0:
            correct_count += 1

    return correct_count / X_test.shape[0]
