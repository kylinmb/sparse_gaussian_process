import numpy as np
from numpy.linalg import cholesky, det, lstsq
from scipy.optimize import minimize


class FullGP_RBFKernel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def rbf_kernel(self, X1, X2, lscale=1.0, sigma=1.0):
        # sqdist = (x1 - x2)^T(x1 - x2) root or base of kernel functions
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        # k(x1, x2) = sigma^2 * exp(-1/(2*lscale^2)(x1-x2)^T(x1-x2))
        return sigma ** 2 * np.exp(- 0.5 / lscale ** 2 * sqdist)

    def negative_log_likelihood(self, theta):
        K = self.rbf_kernel(self.x_train, self.x_train, lscale=theta[0], sigma=theta[1]) + \
            theta[2] ** 2 * np.eye(len(self.x_train))
        L = cholesky(K)
        return np.sum(np.log(np.diagonal(L))) \
               + 0.5 * self.y_train.T.dot(lstsq(L.T, lstsq(L, self.y_train)[0])[0]) \
               + 0.5 * len(self.x_train) * np.log(2 * np.pi)

    def posterior_predictive(self, x_star, l_kernel=1.0, sigma_kernel=1.0, sigma_noise=1e-8):
        K = self.rbf_kernel(self.x_train, self.y_train, l_kernel, sigma_kernel) + sigma_noise ** 2 * np.eye(
            len(self.x_train))
        K_s = self.rbf_kernel(self.x_train, x_star, l_kernel, sigma_kernel)
        K_ss = self.rbf_kernel(x_star, x_star, l_kernel, sigma_kernel)
        K_inv = np.linalg.inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.y_train)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, cov_s

    def train(self):
        return minimize(self.negative_log_likelihood, np.random.rand(3, ), method='L-BFGS-B', options={'maxiter': 5000, 'disp': True})
