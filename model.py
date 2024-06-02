from torch import nn, optim
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, input_channels: int, num_classes: int):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 64 * 64, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 256x256 -> 128x128
        x = self.pool(F.relu(self.conv2(x)))  # 128x128 -> 64x64
        x = x.reshape(-1, 32 * 64 * 64)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
