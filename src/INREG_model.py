import numpy as np
from math import log
from scipy.optimize import minimize


class INREG_model():
    """
    """

    def __init__(self):
        self.f = None
        self.parameters = None

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
        for t in range(0, T - 1):
            X_t, X_t1 = data[t, :], data[t + 1, :]
            lambda_t = self.f(X_t,t)
            current_log_likelihood += sum(vect_aux(X_t1, lambda_t))
        return current_log_likelihood

    def pred(self, data):
        """

        :param data:
        :return:
        """
        T, M = data.shape
        return np.concatenate([self.f(data[t, :]) for t in range(T)],t).reshape(T, M)

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
        self.f = lambda x,t: self.parameters[0] * x + self.parameters[1]
