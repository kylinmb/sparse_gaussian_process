import matplotlib.pyplot as plt
import numpy as np
import pyGPs

demoData = np.load('../../data/regression_data.npz')
x = demoData['x']
y = demoData['y']
z = demoData['xstar']

model_sparse = pyGPs.GPR_FITC()
model_sparse.setData(x, y)
model_sparse.optimize()
model_sparse.predict(z)
model_sparse.plot()

# Training Error
prediction_x = model_sparse.predict(x)[0]
error_x = np.linalg.norm(prediction_x - y, 2) / np.linalg.norm(y, 2)
print('Training Error: %e' % error_x)

# Spectrum
induction_points = model_sparse.u
K_mm = model_sparse.covfunc.getCovMatrix(induction_points, induction_points, mode='train')[1]
K_mn = model_sparse.covfunc.getCovMatrix(induction_points, x, mode='cross')
K_nm = np.transpose(K_mn)


def bigLambda(KM, KNM, KMN):
    lamb = np.multiply(KNM, np.transpose(np.linalg.solve(KM, KMN)))
    lamb = np.sum(lamb, axis=1)
    lamb = 1 - lamb
    lamb = np.diag(lamb)
    return lamb


sigma = np.identity(np.shape(x)[0])*model_sparse.posterior.sW**2
covariance = np.matmul(np.matmul(K_nm, np.linalg.inv(K_mm)), K_mn) + bigLambda(K_mm, K_nm, K_mn) + sigma
u, s, v = np.linalg.svd(covariance)

x_axis = np.arange(1, covariance.shape[0] + 1)
plt.plot(x_axis, s, '-r')
plt.title('Spectrum for Sparse GP')
plt.xlabel('Dimension')
plt.ylabel('Singular Value')
plt.show()