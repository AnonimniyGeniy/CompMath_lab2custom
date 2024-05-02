import numpy as np
import pandas as pd


def generate_dataset(n_samples=500):
    np.random.seed(42)

    def target_function(x, y):
        return 35 * x + 10 * y - 100

    X = np.random.uniform(low=-100, high=100, size=(n_samples, 2))

    y_true = target_function(X[:, 0], X[:, 1])
    noise = np.random.normal(loc=0, scale=20, size=n_samples)
    y_noisy = y_true + noise

    return X, y_noisy


X, y = generate_dataset()

# export dataset to csv

df = pd.DataFrame(data=np.hstack((X, y.reshape(-1, 1))), columns=["x1", "x2", "y"])
df.to_csv("dataset.csv", index=False)
