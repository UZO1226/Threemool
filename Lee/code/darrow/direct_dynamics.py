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


class TemperatureForcingRegressor(LeakyDirectDeltaRegressor):
    def __init__(self, model, target, state_coefficient, temperature_coefficient, temperature_center):
        super().__init__(model, target, state_coefficient)
        self.temperature_coefficient = temperature_coefficient
        self.temperature_center = temperature_center

    def predict(self, x):
        return (self.model.predict(x) + self.state_coefficient * x[:, self.target]
                + self.temperature_coefficient * (x[:, 6] - self.temperature_center) - x[:, self.target])


class CoupledForcingRegressor:
    def __init__(self, model, target, state_weights):
        self.model = model
        self.target = target
        self.state_weights = np.asarray(state_weights)

    def predict(self, x):
        return self.model.predict(x) + x[:, :5] @ self.state_weights - x[:, self.target]
