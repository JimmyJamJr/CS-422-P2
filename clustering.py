import numpy as np


def K_Means(X: np.ndarray, K: int, mu: np.ndarray) -> np.ndarray:
    assert K > 0
    # If mu not given, randomly choose K data points
    if mu.size == 0:
        assert X.shape[0] >= K
        mu = X[np.random.choice(X.shape[0], K, replace=False)]

    while True:
        clusters = [[] for i in range(mu.shape[0])]
        for i in range(X.shape[0]):
            distances = [np.linalg.norm(X[i] - mu[j]) for j in range(mu.shape[0])]
            clusters[distances.index(min(distances))].append(X[i])

        new_mu = np.array([np.mean(i, axis=0) for i in clusters])
        if np.array_equal(new_mu, mu):
            return mu
        mu = new_mu