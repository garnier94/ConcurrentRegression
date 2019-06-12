
import  numpy as np
from INREG_model import INREG_model


class Concurrent_INREG(INREG_model):
    def __init__(self, weight=1, season=100, **kwargs):
        INREG_model.__init__(self, **kwargs)
        self.weight = weight
        self.season = season
        self.g = None

        def aux(X, t):
            gX = self.g([X[hist] / (self.season[t-self.horizon-hist] * weight) for hist in range(self.history)])
            weighted_sum = gX.dot(self.weight)
            return self.season[t] * self.weight * gX / weighted_sum

        self.f = aux

    def estim_weight(self,X, n):
        """Estimation de poids normalisé"""
        weight = X[:n,:].sum(axis=0)
        self.weight = weight /sum(weight)


class Concurrent_INAR(Concurrent_INREG):
    """ INAR(1) model"""

    def __init__(self,   **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        self.parameters = np.array([1 for i in range(self.history)] + [0])
        self.bound = [(1E-3,1E3) for i in range(self.history+1)]
        self.g = lambda x: sum( [self.parameters[hist] * x[hist]  for hist in range(self.history)])+ self.parameters[self.history]


class Concurrent_log_INAR(Concurrent_INREG):
    """ INAR(1) model"""

    def __init__(self, **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        self.parameters = np.array([1, 0])
        self.bound = ((1e-10, 1000), (0., 100))
        self.g = lambda x: self.parameters[0] * np.log(1 + x) + self.parameters[1]


