import numpy as np
import pandas as pd
import random
import os


def gen_target_function(x, y):
    def target_function(x_p, y_p):
        return x * x_p + y * y_p

    return target_function


def generate_dataset(target_function, n_samples=500):
    np.random.seed(42)

    X = np.random.uniform(low=-100, high=100, size=(n_samples, 2))

    y_true = target_function(X[:, 0], X[:, 1])
    noise = np.random.normal(loc=0, scale=20, size=n_samples)
    y_noisy = y_true + noise

    return X, y_noisy


def main():
    n_datasets = int(input("Enter the number of datasets to generate: "))
    with open("coefficients.txt", "w") as f:
        for i in range(n_datasets):
            x = round(random.uniform(-100, 100), 1)
            y = round(random.uniform(-100, 100), 1)
            target_function = gen_target_function(x, y)
            X, y_noisy = generate_dataset(target_function)
            df = pd.DataFrame(data=np.hstack((X, y_noisy.reshape(-1, 1))), columns=["x1", "x2", "y"])
            if not os.path.exists('datasets'):
                os.makedirs('datasets')
            df.to_csv(os.path.join("datasets", f"dataset_{i}.csv"), index=False)
            f.write(f"For dataset_{i}, coefficients are: {x}, {y}\n")
            print(f"Dataset {i} generated with target function {x} * x + {y} * y")


if __name__ == "__main__":
    main()
