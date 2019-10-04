import numpy as np
from models.GPR_ARD_Kyli import GPRK


def function(input_val):
    return np.sin(input_val)


def sample_data(n_total):
    x = np.random.rand(n_total, 1)
    y = function(x) + 0.1 * np.random.randn(n_total, 1)
    return x, y


def test_gpr():
    x, y = sample_data(200)

    # training data
    n_tr = 100
    x_tr = x[1:n_tr, :]
    y_tr = y[1:n_tr, :]

    # test data
    x_test = x[n_tr:, :]
    y_test = y[n_tr:, :]

    model = GPRK(x_tr, y_tr)
    model.train()
    y_predicted = model.eval(x_test)
    y_mean = y_predicted[0]
    y_var = y_predicted[1]
    error = np.linalg.norm(y_mean - y_test, 2)/np.linalg.norm(y_test, 2)
    print('Error: %e' % error)


test_gpr()
