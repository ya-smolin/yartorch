# Modifying the Node class to handle different types for power in __pow__ method
import time

import cv2
import graphviz
import numpy as np
from PIL.Image import Image
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

from yarograd.node import Node

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

def draw_dot(val, grad_check):
    g = graphviz.Digraph()
    seen = set()

    def build(v):
        if v not in seen:
            seen.add(v)
            line1 = f"{v.label}={v.get_label_p()}={round(v.data, 3)}"
            tllabel = f"dl_d{v.label}"
            ttlabel=""
            if tllabel in grad_check:
                ttlabel = grad_check[tllabel]
            dc_dp_labels = [f"d{v.label}/d{p.label}={np.round(v.dc_dp[i], 3)}" for i, p in enumerate(v.parents)]
            if dc_dp_labels:
                dc_dp_labels = str(dc_dp_labels).replace("'", "")
            else:
                dc_dp_labels = ""
            g.node(str(id(v)), label=f"{line1}\n{dc_dp_labels}{v.dc_dp if v.op is None else ''}\n{tllabel}={v.grad}\nT{tllabel}={ttlabel}", shape="box")
            for p in v.parents:
                g.edge(str(id(p)), str(id(v)))
                build(p)

    build(val)
    return g

def visualiza_graph():
    g = draw_dot(l, grad_check)
    g.format = 'png'  # Set format to PNG
    png_bytes = g.pipe(format='png')  # Get PNG byte representation
    nparr = np.frombuffer(png_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imshow('Graph Image', img)
    cv2.waitKey(1)

def generate_dataset():
    import numpy as np

    num_samples_per_class = 1000
    negative_samples = np.random.multivariate_normal(
        mean=[0, 3],
        cov=[[1, 0.5], [0.5, 1]],
        size=num_samples_per_class)
    positive_samples = np.random.multivariate_normal(
        mean=[3, 0],
        cov=[[1, 0.5], [0.5, 1]],
        size=num_samples_per_class)

    inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

    targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                         np.ones((num_samples_per_class, 1), dtype="float32")))

    return inputs, targets

def check_torch(w0, b0, x0, y0):
    import torch

    # Initialize x as a tensor with gradient tracking
    w1 = torch.tensor(w0, requires_grad=True)
    b1 = torch.tensor(b0, requires_grad=True)
    x1 = torch.tensor(x0)
    x1.requires_grad = True
    x1.retain_grad()
    y1 = torch.tensor(y0)
    y1.requires_grad = True
    y1.retain_grad()
    u = w1 * x1

    u.retain_grad()
    z = u + b1

    z.retain_grad()
    a = torch.relu(z)

    a.retain_grad()
    lm = a - y1

    lm.retain_grad()
    l = lm ** 2


    l.retain_grad()
    # Perform backpropagation to calculate dl/dx
    l.backward()

    # dl/dx will be stored in x.grad
    dl_dw1, dl_b1 = w1.grad, b1.grad
    return {"dl_dw1" : dl_dw1, "dl_b0": dl_b1, "dl_dx1":x1.grad, "dl_dy1":y1.grad, "dl_du":u.grad, "dl_dz":z.grad, "dl_da":a.grad, "dl_dlm":lm.grad, "dl_dl":l.grad}



X, Y = generate_dataset()

# normalize dataset
# X = (X - X.mean()) / X.std()
# Y = (Y - Y.mean()) / Y.std()

w01 = 1.0
w02 = 1.0
b01 = 1.0
learnrate = 0.0001
epoch = 4

vv = None
for i in range(epoch):
    for (x, y), c in zip(X, Y):
        c = c[0]
        # Rebuild the computational graph
        w1 = Node(w01, label="w1")
        w1.is_require_grads = True
        w2 = Node(w02, label="w2")
        w2.is_require_grads = True

        b1 = Node(b01, label="b1")
        b1.is_require_grads = True
        x1 = Node(x, label="x1")
        y1 = Node(y, label="y1")
        c1 = Node(c, label="c")

        u1 = w1 * x1
        u1.label = "u1"

        u2 = w2 * y1
        u2.label = "u2"

        u = u1 + u2
        u.label = "u"

        z = u + b1
        z.label = "z"

        # a = z.relu()
        # a.label = "a"

        lm = z - c1
        lm.label = "lm"

        l = lm ** 2
        l.label = "l"

        l.backward()

        w01 = w01 - w1.grad * learnrate
        w02 = w02 - w2.grad * learnrate
        b01 = b01 - b1.grad * learnrate
        print(f"loss:{l.data:.5f}", f"w1:{w01:.5f}", f"w2:{w02:.5f}", f"b0:{b01:.5f}")

        #print("==")
        #print(w1.grad, " ", b1.grad)
        #grad_check = check_torch(float(w0), float(b0),  float(x), float(y) )
        #w1grad, b1grad = grad_check["dl_dw1"], grad_check["dl_b0"]
        #print(w1grad.item(), " ", b1grad.item())
        #print("==")

    import matplotlib.pyplot as plt
    # clear previous plot
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0])
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    x2 = lambda xx: (0.5-b01) / w02 - (w01 / w02) * xx
    # plot this function
    xx = np.linspace(min_x, max_x, 100)
    yy = x2(xx)
    plt.plot(xx, yy, '-r')
    plt.draw()
    plt.pause(1)

X_new = np.zeros((2000, 3))
for i in range(X.shape[0]):
    x1, x2 = X[i]
    x_new = w01*x1 + w02*x2 + b01
    X_new[i] = x1, x2, x_new

plt.clf()
# Create the initial plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.where(Y < 0.5, 'r', 'b')
# make a vector of colors, delete dims unnecessary
colors = np.squeeze(colors)
ax.scatter(X_new[:, 0], X_new[:, 1],  X_new[:, 2], c=colors, depthshade=True)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Value')
# ax.set_zlim(0, 1)  # Set Z-axis range
ax.set_title('3D Surface Plots')
# plt.scatter(X_new[:, 0], X_new[:, 1], c=Y[:, 0])
# plt.draw()

#draw histogram from Y
plt.figure()
plt.hist(X_new[:, 2], bins=16)
plt.show()