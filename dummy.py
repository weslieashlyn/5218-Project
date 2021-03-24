# does not work yet

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.autograd import Variable
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset,DataLoader

import PIL
from PIL import Image
from numpy import asarray

import numpy as np
from sklearn.dummy import DummyClassifier

# is_cuda = False
# if torch.cuda.is_available():
#     is_cuda = True

# simple_transform = transforms.Compose([transforms.Resize((120,120))
#                                        ,transforms.ToTensor()
#                                       ])

# dataset = ImageFolder('flowers', simple_transform)

# load the images
image1 = Image.open('flowers\daisy\5547758_eea9edfd54_n.jpg')
image2 = Image.open('flowers\dandelion\7355522_b66e5d3078_m.jpg')
image3 = Image.open('flowers\rose\12240303_80d87f77a3_n.jpg')
image4 = Image.open('flowers\sunflower\6953297_8576bf4ea3.jpg')
image5 = Image.open('flowers\tulip\10791227_7168491604.jpg')
# convert images to numpy array
frame1 = asarray(image1)
frame2 = asarray(image2)
frame3 = asarray(image3)
frame4 = asarray(image4)
frame5 = asarray(image5)

dummy_clf = DummyClassifier(strategy = "stratified")
dummy_clf.fit(frame1, frame2, frame3, frame4, frame5)
dummy_clf.predict(frame1)

# print(len(dataset), len(dataset.classes))
