import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize


def rbf_kernel(X1, X2, lscale=1.0, sigma=1.0):
    """
    Isotropic squared exponential kernel. Computes a covariance
    matrix from points in x1 and x2
    :param x1: Array of m points (m x d)
    :param x2: Array of n points (n x d)
    :param lscale: length scale
    :param sigma: standard deviation
    :return: Covariance Matrix (m x n)
    """
    # sqdist = (x1 - x2)^T(x1 - x2) root or base of kernel functions
    sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
    # k(x1, x2) = sigma^2 * exp(-1/(2*lscale^2)(x1-x2)^T(x1-x2))
    return sigma**2 * np.exp(- 0.5 / lscale**2 * sqdist)


# Example of Prior
X = np.arange(-5, 5, 0.2).reshape(-1, 1)

mu = np.zeros(X.shape)
cov = rbf_kernel(X, X)

samples = np.random.multivariate_normal(mu.ravel(), cov, 3)

plt.plot(X, samples[0])
plt.plot(X, samples[1])
plt.plot(X, samples[2])
plt.title('Prior')
plt.show()


def posterior_predictive(X_s, X_train, Y_train, l_kernel=1.0, sigma_kernel=1.0, sigma_noise=1e-8):
    K = rbf_kernel(X_train, X_train, l_kernel, sigma_kernel) + sigma_noise**2 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train, X_s, l_kernel, sigma_kernel)
    K_ss = rbf_kernel(X_s, X_s, l_kernel, sigma_kernel)
    K_inv = np.linalg.inv(K)

    mu_s = K_s.T.dot(K_inv).dot(Y_train)
    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s


X_train = np.array([-4, -3, -2, -1, 1]).reshape(-1, 1)
Y_train = np.sin(X_train)

mu_s, cov_s = posterior_predictive(X, X_train, Y_train)
samples = np.random.multivariate_normal(mu_s.ravel(), cov_s, 3)
plt.plot(X, samples[0])
plt.plot(X, samples[1])
plt.plot(X, samples[2])
plt.show()


def negative_log_likelihood(theta, X_train, Y_train):
    K = rbf_kernel(X_train, X_train, lscale=theta[0], sigma=theta[1]) + theta[2]**2 * np.eye(len(X_train))
    return 0.5 * (np.log(np.linalg.det(K))
                  + Y_train.T.dot(np.linalg.inv(K)).dot(Y_train)
                  + len(X_train) * np.log(2*np.pi))


res = minimize(negative_log_likelihood, np.random.rand(3,), args=(X_train, Y_train), method='L-BFGS-B')