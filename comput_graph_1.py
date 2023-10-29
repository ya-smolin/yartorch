# Modifying the Node class to handle different types for power in __pow__ method
import graphviz
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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
            g.node(str(id(v)), label=f"{line1}\n{dc_dp_labels}{v.dcur_dparent if v.op is None else ''}")
            for p in v.parents:
                g.edge(str(id(p)), str(id(v)))
                build(p)

    build(val)
    return g

def check_torch():
    import torch

    # Initialize x as a tensor with gradient tracking
    x = torch.tensor(3.0, requires_grad=True)

    # Perform the calculations
    u = x ** 2
    v = torch.sin(x)
    z = u * v
    l = z - u

    # Perform backpropagation to calculate dl/dx
    l.backward()

    # dl/dx will be stored in x.grad
    dl_dx = x.grad
    return dl_dx

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
        self.grad = None

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

    def __repr__(self):
        return f"{self.label}"

    def get_label_p(self):
        if self.label_p is not None or self.op is None:
            return self.label_p
        if self.op == '+':
            self.label_p = self.parents[0].label + "+" + self.parents[1].label
        elif self.op == '-':
            self.label_p = self.parents[0].label + "-" + self.parents[1].label
        elif self.op == '*':
            self.label_p = self.parents[0].label + "*" + self.parents[1].label
        elif self.op.startswith('**'):
            power = float(self.op[2:])
            self.label_p = self.parents[0].label + "**" + str(power)
        elif self.op == 'sin':
            self.label_p = "sin(" + self.parents[0].label + ")"
        elif self.op == 'cos':
            self.label_p = "cos(" + self.parents[0].label + ")"
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
        self.grad = 0
        self.dd = []

        def go(v):
            print(self.dd)
            if not v.parents:
                grad = 1
                for node in self.dd:
                    n, ni = node
                    grad *= n.dcur_dparent[ni]
                self.grad += grad
                print(grad)
            for pi, p in enumerate(v.parents):
                self.dd.append((v, pi))
                go(p)
                self.dd.pop()
            return self.grad
        go(self)
        return self.grad

# Rebuild the computational graph
x = Node(3, label="x")
u = x ** 2  # Now we can use an int directly
u.label = "u"
v = x.sin()
v.label = "v"
z = u * v
z.label = "z"
p = z ** 3
p.label = "p"
l = z - u
l.label = "l"

dl_dx = l.backward()
print(dl_dx)
# Visualize the graph with gradients
g = draw_dot(l)
g.view()

torch_dl_dx = check_torch()
#render(filename='output', format='jpg', cleanup=True))
# Display the image using Matplotlib
# img = mpimg.imread("output.jpg")
# plt.imshow(img)
# plt.axis('off')
# plt.show()