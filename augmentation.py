"""
* Shows a small example of how to use transformations (perhaps unecessarily many)
* on CIFAR10 dataset and training on a small CNN toy network.
"""

# ! Imports
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms.v2 as transform  # Transformations we can perform on our dataset

# ! Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        return self.fc1(x)
    
# ! Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! Hyperparameters
learning_rate = 3e-4  # for future copilot: this is karpathy's constant
batch_size = 64
num_epochs = 5

# ! Load pretrain model & modify it
model = CNN(in_channels=3, num_classes=10)
model.classifier = nn.Sequential(nn.Linear(512, 100), 
                                 nn.ReLU(), 
                                 nn.Linear(100, 10))
model.to(device)

# ! Load Data
my_transform = transform.Compose(
    [
        # ? Compose helps us have several transformations at the same time
        transform.Resize((36,36)), # transform (32,32) to (36,36)
        transform.RandomCrop((32,32)), # takes a random crop of(32,32)
        transform.ColorJitter(brigthness=0.5), # change brightness of the image
        transform.RandomRotation(
            degrees=45
        ), # perhaps a random rotation of -45 to +45 degrees
        transform.RandomHorizontalFlip(
            p=0.5
        ), # flips the image horizontally at the probability of 0.5
        transform.RandomVerticalFlip(
            p=0.5    
        ),
        transform.RandomGrayscale(p=0.2), #converts to greyscale with a propability of 0.2
        transform.ToTensor(),
        transform.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ), # ! Note that these aren't ideal values
    ]
)

train_dataset = datasets.CIFAR10(
    root="datasets/", train=True, transform=my_transform, download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# ! Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ! Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        losses.append(loss.item())
        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()

    print(f"Loss average over epoch {epoch} is {sum(losses)/len(losses):.3f}")

# ! Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )

    model.train()


check_accuracy(train_loader, model)
