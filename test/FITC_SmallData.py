import matplotlib.pyplot as plt
from models.FITC_Kyli import FITC
import numpy as np
from scipy.io import loadmat

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

model = FITC(x_train, y_train)
model.train()


k = model.get_kernel_matrix()
u, s, v = np.linalg.svd(k)

x_axis = np.arange(1, k.shape[0]+1)
plt.plot(x_axis, s, 'r-')
plt.xlim(0, 40)
plt.title('Sparse GP')
plt.xlabel('Dimension')
plt.ylabel('Singular Value')
plt.show()

# # Training Error and Test Error
y_predicted_train = model.eval(x_train)
y_mean_train = y_predicted_train[0]
error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)
print('Train Error: %e' % error_train)

y_predicted_test = model.eval(x_test)
y_mean_test = y_predicted_test[0]
error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)
print('Test Error: %e' % error_test)

