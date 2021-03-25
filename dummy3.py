import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier

# Model to device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomRotation(0.2),
                                transforms.ToTensor(),
                                transforms.Resize((80,80))
                               ])

dataset = torchvision.datasets.ImageFolder(root = 'flowers',
                                           transform = transform)
print("No of Classes: ", len(dataset.classes))

train, val = torch.utils.data.random_split(dataset, [3000, 1323])

train_loader = torch.utils.data.DataLoader(dataset = train,
                                           batch_size = 32, 
                                           shuffle = True)

val_loader = torch.utils.data.DataLoader(dataset = val,
                                         batch_size = 32, 
                                         shuffle = True)


dummy_clf = DummyClassifier(strategy = "uniform")
dummy_clf.fit(train_loader, val_loader)
dummy_clf.predict(val_loader)
