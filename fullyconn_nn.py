"""
* A simple walkthrough of how to code a fully connected neural network
* using the PyTorch library. For demonstration we train it on the very
* common MNIST dataset of handwritten digits. In this code we go through
* how to create the network as well as initialize a loss function, optimizer,
* check accuracy and more.
"""
# ! Imports
import torch
import torch.nn.functional as F  # ? Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # ? Standard datasets
import torchvision.transforms as transforms  # ? Transformations we can perform on our dataset for augmentation
from torch import optim  # ? For optimizers like SGD, Adam, etc.
from torch import nn  # ? All neural network modules (like nn.linear or nn.Conv)
from torch.utils.data import (
    DataLoader,
)  # ? Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # ? For nice progress bar!

# * Here we create our simple neural network. For more details here we are subclassing and
# * inheriting from nn.Module, this is the most general way to create your networks and
# * allows for more flexibility. I encourage you to also check out nn.Sequential which
# * would be easier to use in this scenario but I wanted to show you something that
# * "always" works and is a general approach.
# ! Create fully-connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        """
        * Here we define the layers of the network. We create two fully connected layers

        * Parameters:
            * input_size: the size of the input, in this case 784 (28x28)
            * num_classes: the number of classes we want to predict, in this case 10 (0-9)

        """
        super(NN, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
    def forward(self, x):
        """
        * x here is the mnist images and we run it through fc1, fc2 that we created above.
        * we also add a ReLU activation function in between and for that (since it has no parameters)
        * I recommend using nn.functional (F)

        * Parameters:
            * x: mnist images

        * Returns:
            * out: the output of the network
        """
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ! Set device
# ? Set device cuda for GPU if it's available otherwise run on the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! Hyperparameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 3

# ! Load data
train_dataset = datasets.MNIST(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
) # transforming data from numpy to tensor, so that it can run in pyTorch
test_dataset = datasets.MNIST(
    root="dataset/", train=False, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # makes sure we do not have the same batch in multiple continous epochs
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# ! Initialize Network
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# ! Loss and Optimizer

# ! Train Network

# ! Check accuracy on training & test