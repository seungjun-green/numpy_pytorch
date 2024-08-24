import numpy as np

class MeanSquaredError:
    @staticmethod
    def loss(predicted, target):
        return np.mean((predicted - target) ** 2)

    @staticmethod
    def loss_derivative(predicted, target):
        return 2 * (predicted - target) / target.size
    
class BinaryCrossEntropy:
    @staticmethod
    def loss(predicted, target):
        return -np.mean(target * np.log(predicted) + (1 - target) * np.log(1 - predicted))

    @staticmethod
    def loss_derivative(predicted, target):
        return (predicted - target) / (predicted * (1 - predicted))
    
    
class CrossEntropy:
    @staticmethod
    def loss(predicted, target):
        return -np.mean(target * np.log(predicted))

    @staticmethod
    def loss_derivative(predicted, target):
        return predicted - target
    