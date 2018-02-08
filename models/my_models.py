"""
based on https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import torch.nn as nn
import torch.nn.functional as F
import torch

class SimplestNet(nn.Module):
    def __init__(self):
        super(SimplestNet, self).__init__()
        self.dropout = nn.Dropout(p=0.2)
        self.batch1 = nn.BatchNorm2d(2)
        self.conv1 = nn.Conv2d(2, 8, kernel_size=5,stride=2)
        self.batch2 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3,stride=1)
        self.batch3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(2048 + 1, 512)
        self.batch4 = nn.BatchNorm2d(512)
        self.fc2 = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, angles):
        x = self.batch1(x)
        x = self.pool(F.relu(self.batch2(self.conv1(x))))
        #x = self.dropout(x)
        x = self.pool(F.relu(self.batch3(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = torch.cat((x, angles), 1)
        x = self.dropout(x)
        x = F.relu(self.batch4(self.fc1(x)))
        x = self.fc2(x)
        #x = self.sig(x)
        return x

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()

        #self.batch1 = nn.BatchNorm2d(16)
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 9 * 9 + 1, 1024)
        self.dropout = nn.Dropout(p = 0.1)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x, angles):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        #x = self.batch1(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, angles), 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.sig(x)
        return x

class Leak(nn.Module):
    def __init__(self, model):
        super(Leak, self).__init__()
        self.VGG = model
        self.dense = nn.Sequential(
            nn.Linear(3, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x, angles):
        x = self.VGG(x, angles)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, angles), 1)
        x = self.dense(x)
        return x