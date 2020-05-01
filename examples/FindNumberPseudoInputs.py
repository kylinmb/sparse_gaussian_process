import matplotlib.pyplot as plt
from models.SparseGP_ARDParams import SparseGPARD
import numpy as np
from scipy.io import loadmat
import pandas as pd
import tensorflow as tf

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

# try different number of pseudo inputs
df = pd.DataFrame(columns=['Number Pseudo Inputs', 'Training Error', 'Test Error'])
for i in range(300, 401, 5):
    try:
        print('Number of Pseudo Inputs: %i' % i)
        model = SparseGPARD(x_train, y_train, i)
        model.train()
        y_predicted_train = model.predict(x_train)
        y_mean_train = y_predicted_train[0]
        error_train = np.linalg.norm(y_mean_train - y_train, 2) / np.linalg.norm(y_train, 2)
        print('Train Error: %e' % error_train)

        y_predicted_test = model.predict(x_test)
        y_mean_test = y_predicted_test[0]
        error_test = np.linalg.norm(y_mean_test - y_test, 2) / np.linalg.norm(y_test, 2)
        print('Test Error: %e' % error_test)

        df = df.append({'Number Pseudo Inputs': i, 'Training Error': error_train, 'Test Error': error_test},
                       ignore_index=True)
    except tf.errors.InvalidArgumentError as e:
        print('Cholesky Factorization Failed')
        df = df.append({'Number Pseudo Inputs': i, 'Training Error': float('NaN'), 'Test Error': float('NaN')},
                       ignore_index=True)

plt.plot(df['Number Pseudo Inputs'], df['Training Error'])
plt.plot(df['Number Pseudo Inputs'], df['Test Error'])
plt.xlabel('Number of Pseudo Inputs')
plt.ylabel('Error')
plt.legend(['Training Error', 'Test Error'])
plt.title('Training and Test Error vs Number of Pseudo Inputs')
plt.savefig('../images/TrainAndTestErrorNumPseudoInputs.png')
plt.show()
