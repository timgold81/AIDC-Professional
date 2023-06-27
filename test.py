import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
import time

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
    
def test_image(model, image_path):
    # Load the trained model
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    
    # Open the image and convert it to grayscale
    image = Image.open(image_path).convert('L')
    
    # Resize the image and apply the same transformations as during training
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = transform(image)
    
    # Add a batch dimension and move the image to the device
    image = image.unsqueeze(0).to(device)
    
    # Get the model's prediction
    output = model(image)
    pred = output.argmax(dim=1).item()
    
    return pred

# Start time measurement 
start_time=time.time()

image_path = '7.png'
# Set the device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Net().to(device)
prediction = test_image(model, image_path)
print(f'The predicted number is: {prediction}')

#Time calculation:
end_time=time.time()
elapsed_time=end_time-start_time
print(f'Device : {device}')
print(f'Time : {elapsed_time:.2f} seconds ')