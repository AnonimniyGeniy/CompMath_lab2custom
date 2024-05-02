import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv("dataset.csv")

X = data[["x1", "x2"]].values
y = data["y"].values

plt.figure(figsize=(10, 7))
plt.scatter(X[:, 0], y, label="x1")
plt.scatter(X[:, 1], y, label="x2")
plt.xlabel('X')
plt.ylabel('Y')
plt.show()