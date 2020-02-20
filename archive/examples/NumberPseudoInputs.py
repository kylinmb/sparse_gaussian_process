import GPy
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

# examples data
n_test = 100
x_test = x[n_tr:n_tr + n_test, :]
y_test = y[n_tr:n_tr + n_test, :]

# model
m_full = GPy.models.GPRegression(x_train, y_train)
m_full.optimize('bfgs')

# Full GP - Training Error and Test Error
y_predicted_train = m_full.predict(x_train)
y_mean_train = y_predicted_train[0]
error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)
print('Full Train Error: %e' % error_train)

y_predicted_test = m_full.predict(x_test)
y_mean_test = y_predicted_test[0]
error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)
print('Full Test Error: %e' % error_test)

# Sparse
for i in range(50, 100):
    Z = np.random.rand(i, x_test.shape[1])
    m_sparse = GPy.models.SparseGPRegression(x_train, y_train, Z=Z)
    m_sparse.optimize('bfgs')

    y_predicted_train = m_sparse.predict(x_train)
    y_mean_train = y_predicted_train[0]
    s_error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)

    y_predicted_test = m_sparse.predict(x_test)
    y_mean_test = y_predicted_test[0]
    s_error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)

    print('Step: {0} Full Train Error: {1}'.format(i, s_error_train))
    print('Step: {0} Full Test Error: {1}'.format(i, s_error_test))

    plt.plot(i, s_error_train, 'ro', label='Sparse Train Error')


x_plot = np.arange(50, 100)
train_error_plot = np.ones(50) * error_train
test_error_plot = np.ones(50) * error_test
plt.plot(x_plot, train_error_plot, 'k-', label='Full GP Training Error')
plt.xlim(50, 100)
# plt.legend()
plt.show()