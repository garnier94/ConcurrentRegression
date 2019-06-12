import numpy as np
from math import log
from scipy.optimize import minimize


class INREG_model():
    """
    """

    def __init__(self, horizon=1, history=1):
        self.f = None
        self.parameters = None
        self.horizon = horizon
        self.history = history

    def log_likelihood(self, data):
        """
        @param : data is a TxM matrices containing the data of differents series
        """

        T, M = data.shape

        def aux(x, l):
            if l != 0:
                return l - x * log(l)
            else:
                return x

        vect_aux = np.vectorize(aux)
        current_log_likelihood = 0
        for t in range(self.history - 1, T - self.horizon):
            X_t, X_t1 = [data[t - i, :] for i in range(self.history)], data[t + self.horizon, :]
            lambda_t = self.f(X_t, t)
            current_log_likelihood += sum(vect_aux(X_t1, lambda_t))
        return current_log_likelihood

    def pred(self, data):
        """

        :param data:
        :return:
        """
        T, M = data.shape
        return np.concatenate([self.f([data[t - hist, :] for hist in range(self.history)], t) for t in
                               range(self.history - 1, T)]).reshape(T-self.history + 1, M)

    def fit(self, mat, **kwargs):
        def aux(params):
            self.parameters = params
            return self.log_likelihood(mat)

        res = minimize(aux, self.parameters, bounds=self.bound, **kwargs)
        self.parameters = res.x


class INAR(INREG_model):
    """ INAR(1) model"""

    def __init__(self):
        self.parameters = np.array([1, 0])
        self.bound = ((1e-10, 10), (0., 100))
        self.f = lambda x, t: self.parameters[0] * x[0] + self.parameters[1]
