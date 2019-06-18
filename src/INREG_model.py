import numpy as np
from math import log
from scipy.optimize import minimize


class INREG_model():
    """
    """

    def __init__(self, horizon=1, history=1,**kwargs):
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
            if x == 0:
                return 0
            elif l != 0:
                return l - x * log(l)
            else:
                return x

        vect_aux = np.vectorize(aux)
        current_log_likelihood = 0
        ###
        old_lambda = [data[self.history - 1 - i, :] for i in range(self.history)]

        for t in range(self.history - 1, T - self.horizon):
            X_t, X_t1 = [data[t - i, :] for i in range(self.history)], data[t + self.horizon, :]
            lambda_t = self.f(X_t, t, old_lambda)
            current_log_likelihood += sum(vect_aux(X_t1, lambda_t))
            old_lambda = [lambda_t] + old_lambda[:-1]
        return current_log_likelihood

    def pred(self, data):
        """

        :param data:
        :return:
        """
        T, M = data.shape

        list_pred = list()
        old_lambda = [data[self.history - 1 - i, :] for i in range(self.history)]
        for t in range(self.history - 1, T):
            new_lambda = self.f([data[t - hist, :] for hist in range(self.history)],t,old_lambda)
            list_pred.append(new_lambda)
            old_lambda = [new_lambda] + old_lambda[:-1]
        return np.concatenate(list_pred).reshape(T-self.history + 1, M)

    def fit(self, mat, **kwargs):
        def aux(params):
            self.parameters = params
            return self.log_likelihood(mat)

        res = minimize(aux, self.parameters, bounds=self.bound, **kwargs)
        self.parameters = res.x


class INAR(INREG_model):
    """ INAR(1) model"""

    def __init__(self, **kwargs):
        INREG_model.__init__(self, **kwargs)
        self.parameters = np.array([1 for i in range(self.history + 1)])
        self.bound = [(1E-6, 1E5) for i in range(self.history + 1)]
        self.f = lambda x, t, old: sum([self.parameters[hist] * x[hist] for hist in range(self.history)])+ self.parameters[self.history]
