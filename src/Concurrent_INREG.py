from math import log
import numpy as np
from INREG_model import INREG_model


def _find_max_intersection(series_1, series_2):
    '''Trouve la plus grande intersection non nulles entre deux séries '''

    n = series_1.shape[0]
    pos_ind = (series_1 > 0) & (series_2 > 0)
    i = 0
    i_max, j_max = 0, 0
    for j in range(n):
        if pos_ind[j]:
            if j - i > j_max - i_max:
                j_max = j
                i_max = i
        else:
            i = j
    return [i_max, j_max]


def _sum_period(X, i, j):
    return sum(X[i:j])


class Concurrent_INREG(INREG_model):
    def __init__(self, weight=1, season=100, **kwargs):
        INREG_model.__init__(self, **kwargs)
        self.weight = weight
        self.season = season
        self.g = None

        def aux(X, t, old):
            gX = self.g([X[hist] / (self.season[t - self.horizon - hist] * weight) for hist in range(self.history)],
                        old)
            weighted_sum = max(1,gX.dot(self.weight))
            return self.season[t] * self.weight * gX / weighted_sum

        self.f = aux

    def estim_weight(self, X):
        M = X.shape[1]
        weight = np.ones(M)
        for ind_j in range(1, M):
            num_csd = 0

            ecart = np.array([_find_max_intersection(X[:, ind_i], X[:, ind_j]) for ind_i in range(ind_j)])
            for ind_i in range(ind_j):

                i_max, j_max = ecart[ind_i, 0], ecart[ind_i, 1]
                if j_max - i_max > 4:
                    S_i = _sum_period(X[:, ind_i], (i_max + 1), (j_max - 1))
                    S_j = _sum_period(X[:, ind_j], (i_max + 1), (j_max - 1))
                    weight[ind_j] += weight[ind_i] * S_j / S_i
                    num_csd += 1
            if num_csd > 0:
                weight[ind_j] = weight[ind_j] / num_csd
        self.weight = weight


class Concurrent_INAR(Concurrent_INREG):
    """ INAR(p) model"""

    def __init__(self, **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        csta = kwargs.get('cst', True)
        self.parameters = np.array([1 for i in range(self.history)])
        self.bound = [(1E-6, 1E5) for i in range(self.history)]
        self.g = lambda x, old: sum([self.parameters[hist] * x[hist] for hist in range(self.history)]) + csta


class Concurrent_log_INAR(Concurrent_INREG):
    """ INAR(p) model"""

    def __init__(self, **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        self.parameters = np.array([1 for i in range(self.history)])
        self.bound = [(1E-6, 1E5) for i in range(self.history)]
        self.g = lambda x, old: sum([self.parameters[hist] * np.log(1 + x[hist]) for hist in range(self.history)]) + 1


class Concurrent_INGARCH(Concurrent_INREG):
    """Concurrent INGARCH models"""

    def __init__(self, **kwargs):
        Concurrent_INREG.__init__(self, **kwargs)
        cst = kwargs.get('cst', True)
        self.parameters = np.array([1 for i in range(self.history * 2)])
        self.bound = [(1E-6, 1E5) for i in range(self.history * 2)]
        self.g = lambda x, old: sum([self.parameters[hist] * x[hist] for hist in range(self.history)]) + cst + sum(
            [self.parameters[self.history + hist] * old[hist] for hist in range(self.history)])
