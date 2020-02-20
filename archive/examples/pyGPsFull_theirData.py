import matplotlib.pyplot as plt
import numpy as np
import pyGPs

demoData = np.load('../../data/regression_data.npz')
x = demoData['x']
y = demoData['y']
z = demoData['xstar']

model_full = pyGPs.GPR()
model_full.getPosterior(x, y)
model_full.optimize(x, y)
model_full.predict(z)
model_full.plot()

# Training Error
prediction_x = model_full.predict(x)[0]
error_x = np.linalg.norm(prediction_x - y, 2) / np.linalg.norm(y, 2)
print('Training Error: %e' % error_x)

# Spectrum
covariance = model_full.covfunc.getCovMatrix(x, x, mode='train')
u, s, v = np.linalg.svd(covariance)

x_axis = np.arange(1, covariance.shape[0] + 1)
plt.plot(x_axis, s, '-r')
plt.title('Spectrum for Full GP')
plt.xlabel('Dimension')
plt.ylabel('Singular Value')
plt.show()

