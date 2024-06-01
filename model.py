from torch import nn, optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_size, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 128 * 128, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.reshape(-1, 32 * 128 * 128)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x