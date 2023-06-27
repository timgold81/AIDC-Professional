import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Define a simple convolutional neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        output = nn.functional.log_softmax(x, dim=1)
        return output

# Define the training function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

# Define the validation function
def validate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += nn.functional.nll_loss(output, target,
                                                reduction='sum').item()
            pred = output.argmax(dim=1,
                                 keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss,
        correct,
        len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# Start time measurement 
start_time=time.time()
# Set the device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set the batch size and number of epochs
batch_size = 64
epochs = 10

# Load the training and validation data
train_data = datasets.MNIST(root='data', train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ]))
test_data = datasets.MNIST(root='data', train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ]))

train_loader = DataLoader(train_data,
                          batch_size=batch_size,
                          shuffle=True)
test_loader = DataLoader(test_data,
                         batch_size=batch_size,
                         shuffle=True)

# Initialize the model and optimizer
model = Net().to(device)
optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0)

# Train the model for the specified number of epochs
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    validate(model, device, test_loader)

# Save the trained model
torch.save(model.state_dict(), "mnist_cnn.pt")

# Time calculation:
end_time=time.time()
elapsed_time=end_time-start_time
print(f'Device : {device}')
print(f'Time : {elapsed_time:.2f} seconds ')