"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch"""

# Import deepxde, tensorflow, matplotlib.pyplot, numpy, sys
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt
import numpy as np
import sys

# Create method to save the results data to the models folder
name = sys.argv[1]
save_path = "./models/"+name+"/"

# “Generate” or import data from the exact solution of the PDE problem
def gen_testdata():
    global xx
    global tt
    global t
    global x
    data = np.load("../burgers_eq/Burgers.npz")
    t, x, exact = data["t"], data["x"], data["usol"].T
    xx, tt = np.meshgrid(x, t)
    X = np.vstack((np.ravel(xx), np.ravel(tt))).T
    y = exact.flatten()[:, None]
    return X, y

# Define computational geometry (interval) and time domain and combine using GeometryXTime
geom = dde.geometry.Interval(-1, 1)
timedomain = dde.geometry.TimeDomain(0, 0.99)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Express the PDE residual of the Burgers Equation
def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=1)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    return dy_t + y * dy_x - 0.01 / np.pi * dy_xx

# Boundary/Initial Conditions
bc = dde.icbc.DirichletBC(geomtime, lambda x: 0, lambda _, on_boundary: on_boundary)
ic = dde.icbc.IC(geomtime, lambda x: -np.sin(np.pi * x[:, 0:1]), lambda _, on_initial: on_initial)

# Define the TimePDE problem
data = dde.data.TimePDE(geomtime, pde, [bc, ic], num_domain=2540, num_boundary=80, num_initial=160)

# Choose the neural network (in this case, 3 hidden layers and a width of 20 neurons per layer)
net = dde.nn.FNN([2] + [20] * 3 + [1], "tanh", "Glorot normal")

# Define a Model 
model = dde.Model(data, net)

# Choose the optimization hyperparameters, such as optimizer (Adam) and learning rate (lr=1e-3)
model.compile("adam", lr=1e-3)

# Train the model (15000 iterations)
losshistory, train_state = model.train(iterations=15000)

# After we train the network using Adam, we continue to train the network using L-BFGS to achieve a smaller loss
model.compile("L-BFGS")
losshistory, train_state = model.train()
dde.saveplot(losshistory, train_state, issave=True, isplot=True)

# Predict the PDE solution at different locations using model.predict
X, y_true = gen_testdata()
y_pred = model.predict(X)
f = model.predict(X, operator=pde)
print("Mean residual:", np.mean(np.absolute(f)))
print("L2 relative error:", dde.metrics.l2_relative_error(y_true, y_pred))
np.savetxt(save_path+"test.dat", np.hstack((X, y_true, y_pred)))

#### Plotting routine ####

global xx 
global tt
global t
global x

x_len = len(x)
t_len = len(t)

fig = plt.subplots()
fig.set_figwidth(8)
fig.set_figheight(4)

plt.contourf(tt,xx,y_pred.reshape(t_len,x_len),levels=25, cmap = 'jet')
plt.xlabel('t')
plt.ylabel('x')
plt.title("u(x,t) Predicted Solution")
plt.colorbar()
plt.savefig(save_path + 'Figure_1.png')
plt.show()