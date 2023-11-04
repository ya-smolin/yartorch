# Modifying the Node class to handle different types for power in __pow__ method
import time

import graphviz
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch

m = 2  # Slope
c = 1  # y-intercept

def draw_dot(val):
    g = graphviz.Digraph()
    seen = set()

    def build(v):
        if v not in seen:
            seen.add(v)
            line1 = f"{v.label}={v.get_label_p()}={round(v.data, 3)}"
            dc_dp_labels = [f"d{v.label}/d{p.label}={np.round(v.dcur_dparent[i], 3)}" for i, p in enumerate(v.parents)]
            if dc_dp_labels:
                dc_dp_labels = str(dc_dp_labels).replace("'", "")
            else:
                dc_dp_labels = ""
            g.node(str(id(v)), label=f"{line1}\n{dc_dp_labels}{v.dcur_dparent if v.op is None else ''}\ndl/d{v.label}={v.grad}")
            for p in v.parents:
                g.edge(str(id(p)), str(id(v)))
                build(p)

    build(val)
    return g


# Define the function
def h(x1, y, w1_, b_):
    return (np.maximum(0, w1_ * x1 + b_) - y) ** 2

# random seed=1
np.random.seed(1)

x10 = 5
y0 = 1

fig = plt.figure(figsize=(6, 6))
def plot_loss_function_local(*args):
    x1_ = args[0]
    y_ = args[1]
    w0, b0 = args[2], args[3]
    # Generate grid for x1 and x2 values
    w1_grid = np.linspace(-1, 6, 50)
    b_grid = np.linspace(-1, 6, 50)
    x1_grid, x2_grid = np.meshgrid(w1_grid, b_grid)

    # Compute h values for the surface
    h_values = h(x1_, y_, x1_grid, x2_grid)

    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.plot_surface(x1_grid, x2_grid, h_values, alpha=0.5, label='h(x1, x2)')
    point, = ax.plot([w0], [b0], [0], marker='o', markersize=8, color='red', label='Current (w0, b0)')
    ax.set_xlabel('w1')
    ax.set_ylabel('b')
    ax.set_zlabel('Value')
    #ax.set_zlim(0, 1)  # Set Z-axis range
    ax.set_title('3D Surface Plots')
    return ax

def plot_loss_function(X, Y):
    # Generate meshgrid for w1 and b1
    w1_values = np.linspace(-1, 6, 100)
    b1_values = np.linspace(-1, 6, 100)
    w1_mesh, b1_mesh = np.meshgrid(w1_values, b1_values)

    # Reshape X and Y for broadcasting
    X_reshaped = X[:, np.newaxis, np.newaxis]
    Y_reshaped = Y[:, np.newaxis, np.newaxis]

    # Calculate loss using broadcasting
    Z = np.sum((np.maximum(0, w1_mesh * X_reshaped + b1_mesh) - Y_reshaped) ** 2, axis=0)

    # Create 3D surface plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()
    ax.plot_surface(w1_mesh, b1_mesh, Z, alpha=0.5, label='h(x1, x2)')
    ax.set_xlabel('w1')
    ax.set_ylabel('b1')
    ax.set_zlabel('Loss')
    ax.set_title('3D Surface Plots')

    #plt.show()
    return ax


def generate_dataset():
    # Step 1: Generate x values
    n_points = 100
    x = np.linspace(0, 10, n_points)

    # Step 2: Compute y values using the linear equation y = mx + c

    y_true = m * x + c

    # Step 3: Add Gaussian noise to y values
    noise_stddev = 1  # Standard deviation of Gaussian noise
    y_noisy = y_true + np.random.normal(0, noise_stddev, n_points)

    # Plotting the synthetic dataset
    # plt.scatter(x, y_noisy, label='Noisy Data', c='blue')
    # plt.plot(x, y_true, label='True Line', c='red')
    # plt.legend()
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.title('Synthetic Dataset for 1D Linear Regression')
    # plt.show()
    return x, y_noisy

def check_torch(w0, b0):
    import torch

    # Initialize x as a tensor with gradient tracking
    w1 = torch.tensor(w0, requires_grad=True)
    b1 = torch.tensor(b0, requires_grad=True)
    x1 = torch.tensor(m)
    y1 = torch.tensor(c)

    l = (torch.relu(w1 * x1 + b1) - y1) ** 2

    # Perform backpropagation to calculate dl/dx
    l.backward()

    # dl/dx will be stored in x.grad
    dl_dw1, dl_b1 = w1.grad, b1.grad
    return dl_dw1, dl_b1


class Node:

    def __init__(self, data, parents=None, op=None, label=""):
        if parents is None:
            parents = []
        self.data = data
        self.children = []
        self.parents = parents
        self.op = op
        self.label = label
        self.label_p = None
        self.dcur_dparent = [] # for leaf nodes
        self.grad = 0  # dl/dc
        self.is_require_grads = False

        # Compute gradients as graph is created
        if self.op is not None:
            if self.op == '+':
                self.dcur_dparent = [1, 1]
            elif self.op == '-':
                self.dcur_dparent = [1, -1]
            elif self.op == '*':
                self.dcur_dparent = [self.parents[1].data, self.parents[0].data]
            elif self.op.startswith('**'):
                power = float(self.op[2:])
                self.dcur_dparent = [power * (self.parents[0].data ** (power - 1))]
            elif self.op == 'sin':
                self.dcur_dparent = [np.cos(self.parents[0].data)]
            elif self.op == 'cos':
                self.dcur_dparent = [-np.sin(self.parents[0].data)]
            elif self.op == 'relu':
                self.dcur_dparent = [1 if self.parents[0].data > 0 else 0]

    def __repr__(self):
        return f"{self.label}"

    def get_label_p(self):
        if self.label_p is not None or self.op is None:
            return self.label_p
        if self.op == '+':
            self.label_p = self.parents[0].label + " + " + self.parents[1].label
        elif self.op == '-':
            self.label_p = self.parents[0].label + " - " + self.parents[1].label
        elif self.op == '*':
            self.label_p = self.parents[0].label + " * " + self.parents[1].label
        elif self.op.startswith('**'):
            power = float(self.op[2:])
            self.label_p = self.parents[0].label + "^" + str(power)
        elif self.op == 'sin':
            self.label_p = "sin(" + self.parents[0].label + ")"
        elif self.op == 'cos':
            self.label_p = "cos(" + self.parents[0].label + ")"
        elif self.op == 'relu':
            self.label_p = "relu(" + self.parents[0].label + ")"
        return self.label_p

    def __pow__(self, power, modulo=None):
        if isinstance(power, Node):
            out = Node(self.data ** power.data, [self, power], f'**{power.data}')
        else:
            out = Node(self.data ** power, [self], f'**{power}')
        self.children.append(out)
        return out

    def __add__(self, other):
        out = Node(self.data + other.data, [self, other], '+')
        self.children.append(out)
        other.children.append(out)
        return out

    def __mul__(self, other):
        out = Node(self.data * other.data, [self, other], '*')
        self.children.append(out)
        other.children.append(out)
        return out

    def __sub__(self, other):
        out = Node(self.data - other.data, [self, other], '-')
        self.children.append(out)
        other.children.append(out)
        return out

    def sin(self):
        out = Node(np.sin(self.data), [self], 'sin')
        self.children.append(out)
        return out

    def cos(self):
        out = Node(np.cos(self.data), [self], 'cos')
        self.children.append(out)
        return out

    def backward(self):
        self.dd = []

        def go(v):
            #print(self.dd)
            if v.is_require_grads:
                grad = 1
                for node in self.dd:
                    n, ni = node
                    grad *= n.dcur_dparent[ni]
                v.grad += grad
                #print(f"{v.label}:", grad)
            for pi, p in enumerate(v.parents):
                self.dd.append((v, pi))
                go(p)
                self.dd.pop()
            return self.grad
        go(self)

    def backward_dl_dc(self):
        self.grad = 1
        def go(v):
            if not v.parents: # leaf nodes
                return
            for pi, p in enumerate(v.parents):
                p.grad = v.dcur_dparent[pi] * v.grad
            for pi, p in enumerate(v.parents):
                go(p)
        go(self)

    def relu(self):
        out = Node(np.maximum(self.data, 0), [self], 'relu')
        self.children.append(out)
        return out

X, Y = generate_dataset()

# normalize dataset
# X = (X - X.mean()) / X.std()
# Y = (Y - Y.mean()) / Y.std()

w0 = 6
b0 = 6

learnrate = 0.001

vv = None
for i in range(1):
    for x, y in zip(X, Y):
        # Rebuild the computational graph
        w1 = Node(w0, label="w1")
        w1.is_require_grads = True
        b1 = Node(b0, label="b1")
        b1.is_require_grads = True
        x1 = Node(x, label="x1")
        y1 = Node(y, label="y1")

        u = w1 * x1  # Now we can use an int directly
        u.label = "u"

        z = u + b1
        z.label = "z"

        a = z.relu()
        a.label = "a"

        lm = a - y1
        lm.label = "lm"

        l = lm ** 2
        l.label = "l"

        l.backward_dl_dc()
        if vv is None:
            g = draw_dot(l)
            g.view()
            vv = 5
        print("==")
        print(w1.grad, " ", b1.grad)
        w1grad, b1grad = check_torch(float(w0), float(b0))

        print(w1grad.item(), " ", b1grad.item())
        print("==")
        w0 = w0 - w1.grad * learnrate
        b0 = b0 - b1.grad * learnrate

        print(f"loss:{l.data:.5f}", f"w0:{w0:.5f}", f"b0:{b0:.5f}")

        # ax = plot_function(x, y, w0, b0)
        # plt.pause(1)
        # plt.draw()
