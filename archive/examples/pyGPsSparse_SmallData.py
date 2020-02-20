import pyGPs
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

np.random.seed(101)

data = loadmat('../../data/pol.mat')
x = data['x']
y = data['y']

# training data
n_tr = 2000
x_train = x[1:n_tr, :]
y_train = y[1:n_tr, :]

# test data
n_test = 100
x_test = x[n_tr:n_tr + n_test, :]
y_test = y[n_tr:n_tr + n_test, :]

# model and covaraince
model = pyGPs.GPR_FITC()
model.setData(x_train, y_train)

Z = np.random.rand(1, 200)
m = pyGPs.mean.Zero()
k = pyGPs.cov.RBF()
model.setPrior(mean=m, kernel=k, inducing_points=Z)
model.optimize()
# u = model.u
# covariance = model.covfunc.getCovMatrix(x_train, x_train, mode='train')
# u, s, v = np.linalg.svd(covariance)

# prediction check
# Training Error and Test Error
y_predicted_train = model.predict(x_train)
y_mean_train = y_predicted_train[0]
error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)
print('Train Error: %e' % error_train)

y_predicted_test = model.predict(x_test)
y_mean_test = y_predicted_test[0]
error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)
print('Test Error: %e' % error_test)
#
# x_axis = np.arange(1, covariance.shape[0] + 1)
# plt.plot(x_axis, s, 'r-')
# plt.title('Full GP')
# plt.xlabel('Dimension'
# plt.ylabel('Singular Value')
# # plt.xlim(0, 80)
# plt.show()