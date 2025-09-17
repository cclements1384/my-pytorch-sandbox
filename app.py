import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt 

# load the MNSIT dataset
"""
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0,.5), (0, .0))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
"""

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

#build the model (feed forward neural network with one hidden layer)
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) #first fully connected layer
        self.fc2 = nn.Linear(128,10) #second fully connected layer

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = SimpleNN()

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss();
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model

train_losses = []
num_epocs = 5
for epoc in range(num_epocs):
    running_loss = 0.0
    for i, (images, labels) in enumerate(trainloader, 0):
        if torch.cuda.is_available():
           images, labels = images.cuda(), labels.cuda()
           model.cuda()
        else:
           model.cpu()

        optimizer.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 200 == 99:
            print(f"[Epoc {epoc+1}, Batch {i+1}, Loss:{running_loss / 100:.4f}")
            running_loss = 0.0
    train_losses.append(running_loss/len(trainloader))

print("Program Complete.")