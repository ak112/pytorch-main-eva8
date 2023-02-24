import torch
import torch.nn as nn
import torch.nn.functional as F

class UltimusBlock(nn.Module):
    def __init__(self, input_channels):
        super(UltimusBlock, self).__init__()
        self.K = nn.Linear(input_channels, 8)
        self.Q = nn.Linear(input_channels, 8)
        self.V = nn.Linear(input_channels, 8)
        self.out = nn.Linear(8, input_channels)
        
    def forward(self, x):
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        
        AM = nn.functional.softmax(torch.matmul(Q, K.t()) / (8**0.5), dim=-1)
        Z = torch.matmul(AM, V)
        out = self.out(Z)
        
        return out

class BaseTransformer(nn.Module):
    def __init__(self):
        super(BaseTransformer, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 48, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.ultimus1 = UltimusBlock(48)
        self.ultimus2 = UltimusBlock(48)
        self.ultimus3 = UltimusBlock(48)
        self.ultimus4 = UltimusBlock(48)
        self.fc = nn.Linear(48, 10)
        
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.relu(self.conv3(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.ultimus1(x)
        x = self.ultimus2(x)
        x = self.ultimus3(x)
        x = self.ultimus4(x)
        x = self.fc(x)
        
        return x