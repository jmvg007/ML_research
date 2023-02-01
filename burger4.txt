import deepxde as dde
import numpy as np
from deepxde.backend import tf
import sys

#name = sys.argv[0]
name = sys.argv[1]

dde.config.set_default_float("float64")
tf.random.set_random_seed(1234)

iterations = 15000
lr = 5.e-4
#interiorpts = [20000,1000,2000,2000]
NuMin = 0.01/np.pi
NuMax = 1/np.pi

save_path = "./burger4_models/"+name+"/"


def save_solution(geomtime, model, filename):
    X = geomtime.uniform_points(50**3)
    y_pred = model.predict(X)
    print("Saving u ...\n")
    np.savetxt(filename + "data.dat", np.hstack((X, y_pred)))


def pde(x, y):
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_xx = dde.grad.hessian(y, x, i=0, j=0)
    
    Nu = x[:, 1:2]

    loss = dy_t + y * dy_x - Nu * dy_xx
    
    return loss

'''
def output_transform_flow(x, y):
    x = x[:, 0:1]
    t = x[:, 2:3]

    u_cond = -tf.sin(np.pi*x) / (1 - t * (1 - x**2))

    u = u_cond * y

    return u
'''

def main():
    geom = dde.geometry.Rectangle([-1, NuMin], [1, NuMax])
    timedomain = dde.geometry.TimeDomain(0, 0.99)
    geomtime = dde.geometry.GeometryXTime(geom, timedomain)

    #uniform_points = geomtime.random_points(interiorpts[0])
    #points = uniform_points

    net = dde.maps.FNN([3] + [32] * 6 + [1], "tanh", "Glorot normal")

    #net.apply_output_transform(output_transform_flow)
    net.apply_output_transform(
        lambda x, y: x[:, 2:3] * (1 - x[:, 0:1] ** 2) * y - tf.sin(np.pi * x[:, 0:1])
    )
    

    data = dde.data.TimePDE(
        geomtime,
        pde,
        [],
        num_domain=20000,    # The number of training points sampled inside the domain.
        num_boundary=0,  # The number of training points sampled on the boundary.
        num_initial=0,   # The number of training points sampled on the initial location.
        #num_test=2**13,  # The number of points sampled inside the domain for testing PDE loss. The testing points for BCs/ICs are the same set of points used for training. If None, then the training points will be used for testing.
        #anchors = points # A Numpy array of training points, in addition to the num_domain and num_boundary sampled points.
    )

    model = dde.Model(data, net)

    model.compile("adam", lr=lr)
    losshistory, train_state = model.train(iterations=iterations)

    model.compile("L-BFGS")
    losshistory, train_state = model.train()

    save_solution(geomtime, model, save_path+"./data/solution_")

    dde.saveplot(losshistory, train_state, issave=True, isplot=True)

if __name__ == "__main__":
    main()
