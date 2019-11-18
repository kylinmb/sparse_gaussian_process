import GPy
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

np.random.seed(101)

data = loadmat('../data/pol.mat')
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
m_full = GPy.models.GPRegression(x_train, y_train)
m_full.optimize('bfgs')
covaraince = m_full.kern.K(x_train, x_train)
u, s, v = np.linalg.svd(covaraince)

# prediction check
# Training Error and Test Error
y_predicted_train = m_full.predict(x_train)
y_mean_train = y_predicted_train[0]
error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)
print('Train Error: %e' % error_train)

y_predicted_test = m_full.predict(x_test)
y_mean_test = y_predicted_test[0]
error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)
print('Test Error: %e' % error_test)

x_axis = np.arange(1, covaraince.shape[0] + 1)
plt.plot(x_axis, s, 'r-')
plt.title('Full GP')
plt.xlabel('Dimension')
plt.ylabel('Singular Value')
plt.xlim(0, 80)
plt.show()
