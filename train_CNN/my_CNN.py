import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 158 * 118, 120)  # Adjusted for 640x480 input
        self.fc2 = nn.Linear(120, 84)
        
        # Separate output heads for minutes, seconds, and tenths of seconds
        self.fc_mm = nn.Linear(84, 60)  # Assuming 60 possible values for minutes
        self.fc_ss = nn.Linear(84, 60)  # Assuming 60 possible values for seconds
        self.fc_d = nn.Linear(84, 10)   # Assuming 10 possible values for tenths of seconds

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 158 * 118)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Separate outputs
        mm = self.fc_mm(x)
        ss = self.fc_ss(x)
        d = self.fc_d(x)
        
        return mm, ss, d

