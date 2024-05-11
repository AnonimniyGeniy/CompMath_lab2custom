import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lab2custom import *

# Загружаем данные
df = pd.read_csv("dataset_0.csv")
X = df[["x1", "x2"]].values
y = df["y"].values
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = MyLinearRegression()

losses = model.fit(X_train, y_train, epochs=50, lr=0.0001)


plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

sorted_indices = np.argsort(y_test)

sorted_y_test = y_test[sorted_indices]
sorted_y_pred = model.predict(X_test)[sorted_indices]

# Строим график
plt.figure(figsize=(10, 5))
plt.plot(sorted_y_test, label="True values")
plt.plot(sorted_y_pred, label="Predicted values")
plt.legend()
plt.show()

print(model.get_weights())
