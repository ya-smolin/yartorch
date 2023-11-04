# In[]:

import matplotlib.pyplot as plt
import torch


def show_img(num_img):
    plt.imshow(X[num_img].numpy())
    plt.show()


X = torch.load("mdata/X.pt")
Y = torch.load("mdata/Y.pt")

XXX = X[15][::4, ::4]

plt.imshow(XXX.numpy())
plt.show()

# TODO: rescale dataset to 7by7
# TODO: mean and std
# TODO: forward pass and loss function
# TODO: backfard pass JustDoIt
# TODO: autograd, etc
# TODO: optimizer
# TODO: metrics
