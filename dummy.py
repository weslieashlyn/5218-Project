# does not work yet
# lots of imports from different methods attmpted, feel free to delete any unused

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,DataLoader

import numpy as np
from sklearn.dummy import DummyClassifier

# import os
import sys
import stat

from matplotlib import image
from matplotlib import pyplot

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# import seaborn as sns

import splitfolders
import tensorflow as tf

import base64


# note: split into train, text, validation folders

splitfolders.ratio("Flowers", output="output", seed=1337, ratio=(0.8, 0.1, 0.1))

# train_split = torchvision.datasets.ImageFolder(root = 'output/train')

# test_split = torchvision.datasets.ImageFolder(root = 'output/test')

# val_split = torchvision.datasets.ImageFolder(root = 'output/val')


# turn images into strings
with open("output/train", "rb") as imageFile:
    tr_str = base64.b64encode(imageFile.read())
print(tr_str)

with open("output/val", "rb") as imageFile:
    val_str = base64.b64encode(imageFile.read())
print(val_str)


# convert to numpy array
X = tf.make_ndarray(tr_str)
# nparray_test = tf.make_ndarray(test_split)
Y = tf.make_ndarray(val_str)

# X = np.array(str)
# Y = np.array(val_split)


# note: dummy.fit(flattened tensor of training split, list of attributes)
# note: dummy.predict(perfroms classification)

# torch.flatten(train_split) # note: gives TypeError: flatten(): argument 'input' (position 1) must be Tensor, not ImageFolder

dummy_clf = DummyClassifier(strategy = "uniform")
dummy_clf.fit(X, Y)
dummy_clf.predict(X)
