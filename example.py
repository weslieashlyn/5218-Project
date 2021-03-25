# functional, very slow

# Loading Libraries
import torch 
import torch.nn as nn
import torchvision 
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# Model to device
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

examples = enumerate(val_loader)
batch_idx, (example_data, example_targets) = next(examples)

import matplotlib.pyplot as plt

fig = plt.figure()
for i in range(6):
  plt.subplot(2,3,i+1)
  plt.tight_layout()
  plt.imshow(example_data[i].numpy().transpose())
  plt.title("Ground Truth: {}".format(example_targets[i]))
  plt.xticks([])
  plt.yticks([])

Accuracies = []


# Custom CNN from Kaggle

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1 = nn.Linear(2*2*1024, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 5)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        out = F.dropout(out, training=self.training)
        out = self.fc3(out)
        return F.log_softmax(out,dim=1)

model = ConvNet().to(device)

# Loss and optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

total_step = len(train_loader)
Loss = []
Acc = []
Val_Loss = []
Val_Acc = []

for epoch in range(5):
    acc = 0
    val_acc = 0
    for i, (images, labels) in enumerate(train_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)
    
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Checking accuracy
        preds = outputs.data.max(dim=1,keepdim=True)[1]
        acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
    acc = acc/len(train_loader.dataset) * 100

    for i, (images, labels) in enumerate(val_loader):
        model.eval()
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        val_loss = criterion(outputs, labels)
        
        # Checking accuracy
        preds = outputs.data.max(dim=1,keepdim=True)[1]
        val_acc += preds.eq(labels.data.view_as(preds)).cpu().sum()
    
    val_acc = val_acc/len(val_loader.dataset) * 100
    
    print("Epoch {} =>  loss : {loss:.2f};   Accuracy : {acc:.2f}%;   Val_loss : {val_loss:.2f};   Val_Accuracy : {val_acc:.2f}%".format(epoch+1, loss=loss.item(), acc=acc, val_loss=val_loss.item(), val_acc=val_acc))
  
    Loss.append(loss)
    Acc.append(acc)

    Val_Loss.append(val_loss)
    Val_Acc.append(val_acc)