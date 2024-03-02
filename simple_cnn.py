"""
* A simple walkthrough of how to code a convolutional neural network (CNN)
* using the PyTorch library. For demonstration we train it on the very
* common MNIST dataset of handwritten digits. In this code we go through
* how to create the network as well as initialize a loss function, optimizer,
* check accuracy and more.
"""
# ! Imports
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!

# ! Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10): # in_channels = 1 (for MNIST)
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = 8,
            kernal_size = 3,
            stride = 1,
            padding = 1,
        )
        self.pool = nn.MaxPool2d(kernal_size = 2, stride = 2)
        self.conv2 = nn.Conv2d(
            in_channel = 8,
            out_channel = 16,
            kernal = 3,
            stride = 1,
            padding = 1,
        )
        self.fc1 = nn.Linear(16*7*7, num_classes)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        return x
    
# ! Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4 # karpathy's constant
batch_size = 64
num_epochs = 3

# ! Load data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# ! Initialize network
model = CNN(in_channels=in_channels, num_classes=num_classes).to(device)

# ! Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        