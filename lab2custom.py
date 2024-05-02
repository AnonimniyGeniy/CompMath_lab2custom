import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class MyLinearRegression(object):
    def __init__(self):
        self.w = None

    def fit(self, X, y, epochs=10, lr=0.0001):
        n, k = X.shape
        if self.w is None:
            np.random.seed(42)
            self.w = np.random.randn(k + 1)

        X_train = np.concatenate((np.ones((n, 1)), X), axis=1)

        losses = []

        for i in range(epochs):
            predictions = self._predict_internal(X_train)
            loss = self.__loss(y, predictions)
            losses.append(loss)
            self.w = self.w - lr * self.get_grad(X_train, y, predictions)

        return losses

    def get_grad(self, X_batch, y_batch, predictions):
        grad_basic = 2 * np.transpose(X_batch) @ (predictions - y_batch) / X_batch.shape[0]
        assert grad_basic.shape == (X_batch.shape[1],), "Градиенты должны быть столбцом из k_features + 1 элементов"
        return grad_basic

    def predict(self, X):
        n, _ = X.shape
        X_ = np.concatenate((np.ones((n, 1)), X), axis=1)
        return self._predict_internal(X_)

    def _predict_internal(self, X):
        return np.dot(X, self.w)

    def get_weights(self):
        return self.w.copy()

    def __loss(self, y, p):
        return np.mean((y - p) ** 2)


