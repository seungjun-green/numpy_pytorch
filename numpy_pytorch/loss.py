import numpy as np

class MeanSquaredError:
    @staticmethod
    def loss(predicted, target):
        return np.mean((predicted - target) ** 2)

    @staticmethod
    def loss_derivative(predicted, target):
        return 2 * (predicted - target) / target.size