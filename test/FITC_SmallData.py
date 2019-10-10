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
# y_predicted = model.eval(x_test)
# y_mean = y_predicted[0]
# error = np.linalg.norm(y_mean - y_test, 2)/np.linalg.norm(y_test, 2)
# print('Error: %e' % error)
#
# k = model.get_kernel_matrix()
# u, s, v = np.linalg.svd(k)
#
# xaxis = np.arange(1, k.shape[0]+1)
# n, bins, patches = plt.hist(s, xaxis, density=True, facecolor='r', alpha=0.75)
# plt.xlim(0, 80)
# plt.title('Full GP')
# plt.xlabel('Dimension')
# plt.ylabel('Singular Value')
# plt.show()
