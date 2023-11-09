import numpy as np
import torch


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
        self.dc_dp = []
        self.grad = 0  # dl/dc
        self.torch_tensor = torch.tensor(data)

        # Compute gradients as graph is created
        if self.op is not None:
            if self.op == '+':
                self.dc_dp = [1, 1]
            elif self.op == '-':
                self.dc_dp = [1, -1]
            elif self.op == '*':
                self.dc_dp = [self.parents[1].data, self.parents[0].data]
            elif self.op.startswith('**'):
                power = float(self.op[2:])
                self.dc_dp = [power * (self.parents[0].data ** (power - 1))]
            elif self.op == 'sin':
                self.dc_dp = [np.cos(self.parents[0].data)]
            elif self.op == 'cos':
                self.dc_dp = [-np.sin(self.parents[0].data)]
            elif self.op == 'relu':
                self.dc_dp = [1 if self.parents[0].data > 0 else 0]

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

    def relu(self):
        out = Node(np.maximum(self.data, 0), [self], 'relu')
        self.children.append(out)
        return out

    def backward(self):
        self.grad = 1

        def go(v):
            if not v.parents: # leaf nodes
                return
            for pi, p in enumerate(v.parents):
                p.grad = v.dc_dp[pi] * v.grad
            for pi, p in enumerate(v.parents):
                go(p)
        go(self)