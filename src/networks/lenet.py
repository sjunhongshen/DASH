from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):

    def __init__(self, num_classes=10):

        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.linear = nn.Linear(84, num_classes)

    def forward(self, x):

        out = self.pool1(F.relu(self.conv1(x)))
        out = self.pool2(F.relu(self.conv2(out)))
        out = F.relu(self.fc2(F.relu(self.fc1(out.flatten(1)))))
        return self.linear(out)
