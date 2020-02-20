import matplotlib.pyplot as plt
from models.FullGP_ARDParams import FullGPARD
import numpy as np
from scipy.io import loadmat
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

data = loadmat('../data/pol.mat')
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

model = FullGPARD(x_train, y_train)
model.train()

# Training Error and Test Error
y_predicted_train = model.predict(x_train)
y_mean_train = y_predicted_train[0]
error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)
print('Train Error: %e' % error_train)

y_predicted_test = model.predict(x_test)
y_mean_test = y_predicted_test[0]
error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)
print('Test Error: %e' % error_test)

k = model.get_kernel_matrix()
u, s, v = np.linalg.svd(k)

x_axis = np.arange(1, k.shape[0] + 1)
plt.plot(x_axis, s, 'r-')
# plt.xlim(0, 80)
plt.title('Full GP')
plt.xlabel('Dimension')
plt.ylabel('Singular Value')
plt.show()
