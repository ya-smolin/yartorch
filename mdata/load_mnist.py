# In[]:
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from fastai.data.external import untar_data, URLs
from fastai.data.transforms import get_image_files


def show_img(num_img):
    plt.imshow(X[num_img].numpy())
    plt.show()


path = untar_data(URLs.MNIST)
path_train = path / "training"
path_test = path / "testing"

files = get_image_files(path_train)
# In[]:
print("loading started")
s = time.perf_counter()
N = len(files)
X = np.zeros((N, 28, 28), dtype=np.uint8)
Y = np.zeros((N, 10), dtype=np.uint8)
for i, f in enumerate(files):
    X[i] = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
    Y[i][int(f.parent.name)] = 1
X = torch.from_numpy(X).float()
meanX = X.mean()
stdX = X.std()
X = (X - meanX) / stdX

print(f"mean: {X.mean()} std:{X.std()}")

Y = torch.from_numpy(Y)
print(f"loading ended: {time.perf_counter() - s:.2f}")

torch.save(X, 'X.pt')
torch.save(Y, 'Y.pt')
