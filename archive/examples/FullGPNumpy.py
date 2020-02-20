import numpy as np
from scipy.io import loadmat
from archive.models.GP_NUMPY import FullGP_RBFKernel

data = loadmat('../../data/pol.mat')
x = data['x']
y = data['y']

# training data
n_tr = 2000
x_train = x[1:n_tr, :]
y_train = y[1:n_tr, :]

# examples data
n_test = 100
x_test = x[n_tr:n_tr + n_test, :]
y_test = y[n_tr:n_tr + n_test, :]

model = FullGP_RBFKernel(x_train, y_train)
res = model.train()
theta = res.x

# Training Error and Test Error
mu_train, sig_train = model.posterior_predictive(x_train, theta[0], theta[1], theta[2])
error_train = np.linalg.norm(mu_train - y_train, 2) / np.linalg.norm(y_train, 2)
print('Train Error: %e' % error_train)

mu_test, sig_test = model.posterior_predictive(x_test, theta[0], theta[1], theta[2])
error_test = np.linalg.norm(mu_test - y_test, 2) / np.linalg.norm(y_test, 2)
print('Test Error: %e' % error_test)