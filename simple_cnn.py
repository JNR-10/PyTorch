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
            out_channels = 8, # arbitary number
            kernel_size = 3,
            stride = 1, # sane convolution (ouput size will change based on these values)
            padding = 1,
            # ? n_output = floor([n_in + 2p -k]/s) + 1
        )
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # ? this will half the dimensions
        self.conv2 = nn.Conv2d(
            in_channels = 8, # needs to be the same as the putput of the previous
            out_channels = 16, # arbitary number
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )
        self.fc1 = nn.Linear(16*7*7, num_classes) # ? because 2 pooling layer with 1 padding = (16-1-1)/2 = 14)
        # Linear(in, out)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        # ? this is a 4 dimension tensor (mini_batch, channel, dim*dim)
        # ? and we would keep mini_batches as it is and flatten the other dimensions
        x = self.fc1(x)
        return x
    
"""
* Function for load & save
"""
def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving Checkpoint")
    torch.save(state, filename)
   
def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"]) 
    # optimizer.load_state_dict(checkpoint["optimizer"]) 
    # ? if you save more things like accuracy or epoch, you will have to take this from the dict as well
"""
* Testing
model = CNN()
x = torch.random.randn(64, 1, 28, 28)
print(model(x).shape)
"""
    
# ! Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ! Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 3e-4 # karpathy's constant
batch_size = 64
num_epochs = 10
load_model = True
# ? once we have saved a model and then we keep load_model = False, it overwrites the previously trained file

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

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

# ! Train network
for epoch in range(num_epochs):
    """ 
    * Model Load and Save Code
    """
    losses = []
    if epoch % 3 == 0: # saving every third epoch
        # you can also have condition on accuracy, if accuracy >= some threshold, save the model
        checkpoint = {'state_dict': model.state_dict()}
        save_checkpoint(checkpoint)
    
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
    
    mean_loss = sum(losses)/len(losses)
    print(f'Loss at epoch {epoch} was {mean_loss: .5f}')
        
# ! Check accuracy on training & test 
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, prediction = scores.max(1)
            num_correct += (prediction == y).sum()
            num_samples += prediction.size(0)
            
    model.train()
    return num_correct / num_samples

print(f"Accuracy on training set: {check_accuracy(train_loader, model)*100:.2f}")
print(f"Accuracy on test set: {check_accuracy(test_loader, model)*100:.2f}")
    