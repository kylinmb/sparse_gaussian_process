import numpy as np
import scipy.optimize
from scipy.io import loadmat


def neg_log_likelihood(x, K):
    a = np.linalg.tensorinv(K)
    (s, d) = np.linalg.slogdet(K)
    return 0.5*(np.dot(x, a) + d)


def kernel(theta, dx):
    k = dx[0][1]
    j = dx[0][2]
    d = np.dot(k, j)
    k = np.divide(k, -2*theta[0]**2)
    k = np.exp(k)
    return theta[1]**2 * k


def log_prob(theta, x, y):
    dx = x[:, None] - x[None, :]
    K = kernel(theta, dx)
    K = np.swapaxes(K, 0, 2)
    return neg_log_likelihood(y, K)


data = loadmat('data/pol.mat')
xs = np.copy(data['xs'][0:100])
# xs = np.swapaxes(xs, 0, 1)
ys = np.copy(data['ys'][0:100])
# ys = np.swapaxes(ys, 0, 1)

theta0 = [.01, .01]
result = scipy.optimize.minimize(log_prob, theta0, args=(xs, ys), method='L-BFGS-B')

