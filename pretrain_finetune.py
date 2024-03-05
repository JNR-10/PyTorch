"""
* Shows a small example of how to load a pretrain model (VGG16) from PyTorch,
* and modifies this to train on the CIFAR10 dataset. The same method generalizes
* well to other datasets, but the modifications to the network may need to be changed.
"""

# ! Imports
import torch
import torchvision
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (
    DataLoader,
)  # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! Hyperparameters
num_classes = 10
learning_rate = 1e-3
batch_size = 1024
num_epochs = 3

# ! Loading pre-trained model and modifying it
# ? if we load the model in the datset it was trained on, we simply need to load and run it
# ? but it is not always the case realistically as we would now do fine-tune on CIFAR-10

# ? So if we chwck the architecture of vgg-16, we see the outpur of the CIFAR-10 at the end would be 1x1
# ! so we would need to remove the last avgpool layer in the model as there is not avg pool req in 1x1
# ? and also the output classes in CIFAR are 10
# ! so we also need to change the classifier code at the end of avgpool
import ssl
# Disable SSL verification (not recommended for production)
ssl._create_default_https_context = ssl._create_unverified_context

# Loading pre-trained model and modifying it
model = torchvision.models.vgg16(pretrained=True)

# ? If you want to do finetuning then set requires_grad = False
# ? Remove these two lines if you want to train entire model,
# ? here it would not change the weights of the unmodified layer, ie. won't backprop entire model.
for param in model.parameters():
    param.requires_grad = False

model.avgpool = nn.Identity()
model.classifier = nn.Sequential(
    # since number of filters from vgg is 512
    nn.Linear(512, 100), 
    nn.ReLU(), 
    nn.Linear(100, num_classes)
)
model.to(device)
# import sys
# print(model)
# sys.exit()
# ! Load Data
train_dataset = datasets.CIFAR10(
    root="dataset/", train=True, transform=transforms.ToTensor(), download=True
)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# ! Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ! Train Network
for epoch in range(num_epochs):
    losses = []

    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
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

    print(f"Cost at epoch {epoch} is {sum(losses)/len(losses):.5f}")

# ! Check accuracy on training & test 

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