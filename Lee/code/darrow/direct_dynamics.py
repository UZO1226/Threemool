"""Predict next concentrations while retaining the existing delta interface."""
import numpy as np


class DirectDeltaRegressor:
    def __init__(self, model, target):
        self.model = model
        self.target = target

    def predict(self, x):
        return self.model.predict(x) - x[:, self.target]


class PersistenceRegressor:
    def predict(self, x):
        return np.zeros(len(x))


class LeakyDirectDeltaRegressor:
    def __init__(self, model, target, state_coefficient):
        self.model = model
        self.target = target
        self.state_coefficient = state_coefficient

    def predict(self, x):
        return self.model.predict(x) + self.state_coefficient * x[:, self.target] - x[:, self.target]
