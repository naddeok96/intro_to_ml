import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import wandb
from tqdm import tqdm

# Initialize wandb
wandb.init(project="mnist-pytorch", entity="naddeok")

# Load MNIST dataset without any normalization
print("Loading MNIST dataset without normalization...")
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
trainloader = DataLoader(trainset, batch_size=1000, shuffle=False)

# Calculate mean and std
print("Calculating mean and std...")
mean = 0.0
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    mean += images.mean(2).sum(0)
mean = mean / len(trainloader.dataset)

var = 0.0
for images, _ in trainloader:
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)
    var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
std = torch.sqrt(var / (len(trainloader.dataset) * 28 * 28))

# Data Preprocessing with calculated mean and std
print("Applying data preprocessing...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Reload MNIST dataset with normalization
print("Reloading MNIST dataset with normalization...")
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=16, shuffle=True)

# Define Neural Network
print("Defining Neural Network...")
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model, loss function, and optimizer
print("Initializing model, loss function, and optimizer...")
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training Loop
print("Starting Training Loop...")
for epoch in range(2):
    print(f"Epoch {epoch+1}")
    pbar = tqdm(enumerate(trainloader), total=len(trainloader), desc=f"Epoch {epoch+1}")
    for i, data in pbar:
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)

        # Log loss and accuracy to wandb
        wandb.log({"loss": loss.item(), "accuracy": accuracy})
        
        # Update tqdm progress bar with fixed number of decimals for loss and accuracy
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Accuracy": f"{accuracy:.4f}"})


print("Finished Training")
