
import  numpy as np
from src.INREG_model import INREG_model


class Concurrent_INREG(INREG_model):
    def __init__(self, weight=1, season=100):
        self.weight = weight
        self.season = season
        self.g = None

        def aux(X):
            gX = self.g(X / self.season)
            weighted_sum = gX.dot(self.weight)
            return self.season * self.weight * gX / weighted_sum

        self.f = aux


class Concurrent_INAR(Concurrent_INREG):
    """ INAR(1) model"""

    def __init__(self, **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        self.parameters = np.array([1, 0])
        self.bound = ((1e-10, 100), (1e-10, 100))
        self.g = lambda x: self.parameters[0] * x + self.parameters[1]


class Concurrent_log_INAR(Concurrent_INREG):
    """ INAR(1) model"""

    def __init__(self, **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        self.parameters = np.array([1, 0])
        self.bound = ((1e-10, 1000), (0., 100))
        self.g = lambda x: self.parameters[0] * np.log(1 + x) + self.parameters[1]