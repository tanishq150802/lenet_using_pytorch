#import torch
import torch.nn as nn

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 6,
            kernel_size = (5,5),
            padding = 0,
            stride = 1
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=(2,2),
            padding=0,
            stride=2
        )
        self.conv2 = nn.Conv2d(
            in_channels=6,
            out_channels=16,
            kernel_size=(5,5),
            stride=1,
            padding=0
        )
        self.pool2=nn.MaxPool2d(
            kernel_size=(2,2),
            stride=2,
            padding=0
        )
        self.l1 = nn.Linear(400, 120)
        self.l2 = nn.Linear(120,84)
        self.l3 = nn.Linear(84,10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = x.view(-1, 400) ## Flatten
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return x
    