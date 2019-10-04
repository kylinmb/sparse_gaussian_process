import matplotlib.pyplot as plt
import pyGPs as pyGP
from scipy.io import loadmat

data = loadmat('data/pol.mat')
xs = data['xs']
ys = data['ys']
x = data['x']
y = data['y']

model = pyGP.GPR_FITC()
model.setData(x, y)
model.optimize()
yu, ys2, fu, fs2, lp = model.predict(xs)

fig = plt.figure()
ax = fig.add_subplot()
ax.scatter(x, y, s=10, c='b', marker='o-', label='Input Data')
ax.scatter(xs, yu, s=10, c='r', marker='o-', label='Predicted Data')
plt.show()
