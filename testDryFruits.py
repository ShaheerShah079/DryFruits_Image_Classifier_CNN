import os
import torch
import glob
import torch.nn as nn
from torch.optim import Adam
from torch.optim import SGD

from torchvision.transforms import transforms
import torchvision
import matplotlib.pyplot as plt


def main():
    # Check if Gpu Available on Pc or not Use for Training Model on GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    testing_path='E:/Shaheer Data/Comsats/Semester7/machineLearning/Assignment2/Data/test'
    transformer = transforms.Compose([
        transforms.Resize((200, 200)),  # Resize image to 200x200 pixels
        transforms.ToTensor(),  # Convert image/Numpy Array to PyTorch tensor and scale pixel values to [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5],
                             # Normalize tensor to have mean 0 and std 1 (scales to [-1, 1]),formula (x-mean)/std
                             [0.5, 0.5, 0.5]),
    ])
    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testing_path,transform=transformer),
        batch_size=32, shuffle=True
    )
    model = torch.load('E:/Shaheer Data/Comsats/Semester7/machineLearning/Assignment2/model2.pth')
    test_accuracy=0
    test_count = len(glob.glob(testing_path + '/**/*.jpg'))

    # Iterate over the test data batches, where 'i' is the batch index, and 'images' and 'label' are the current batch of input images and their corresponding labels/output
    for i, (images, label) in enumerate(test_data):
        # images[0] contain all 32 batch of image and label contain all 32 label correspond to each images

        _,prediction=torch.max(model(images).data,1)
        # prediction = torch.argmax(model(images))
        # print("prediction")
        # print(prediction.numpy())
        # print('label data')
        # print(label.data)

        test_accuracy += int(torch.sum(prediction == label.data))

    print("Total Test Image ",test_count)
    print("Correctly Predicted Test Image ",test_accuracy)
    test_accuracy = test_accuracy / test_count
    print("Test Accuracy Percenatge ",test_accuracy)

class ConvNet(nn.Module):
    def __init__(self, numofclasses):
        super(ConvNet, self).__init__()

        # Input shape= (32,3,200,200)
        # Define the first convolutional layer with 3 input channels RGB image and 12 output channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        # Apply batch normalization to the output of the previous layer with 35 feature channels.
        # This helps to stabilize and accelerate training by normalizing the activations of the layer
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.ReLU1 = nn.ReLU()  # Apply ReLU activation

        # Shape= (32,12,200,200)
        # Define the second convolutional layer with 12 input channels and 25 output channels
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=25, kernel_size=3, stride=1, padding=1)
        self.ReLU2 = nn.ReLU()  # Apply ReLU activation

        # Shape= (32,25,200,200)
        # Define max pooling layer with 2x2 kernel and stride of 2
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Shape= (32,25,100,100)
        self.conv3 = nn.Conv2d(in_channels=25, out_channels=35, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=35)
        self.ReLU3 = nn.ReLU()  # Apply ReLU activation

        # Shape= (32,35,100,100)
        self.conv4 = nn.Conv2d(in_channels=35, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.ReLU4 = nn.ReLU()  # Apply ReLU activation

        # Shape= (32,15,100,100)
        # Define max pooling layer with 2x2 kernel and stride of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Shape= (32,15,50,50)
        # Flatten layer to convert 2D feature maps to 1D tensor
        self.fl = nn.Flatten()

        # Define the first fully connected layer with input features and 200 output features
        self.fc1 = nn.Linear(in_features=50 * 50 * 15, out_features=500)
        self.dropout1 = nn.Dropout(0.5)

        # Define the second fully connected layer with 200 input features and output features equal to numofclasses=10
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(in_features=100, out_features=numofclasses);

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.ReLU1(output)

        output = self.conv2(output)
        output = self.ReLU2(output)

        output = self.pool1(output)

        output = self.conv3(output)
        output = self.bn2(output)
        output = self.ReLU3(output)

        output = self.conv4(output)
        output = self.ReLU4(output)

        output = self.pool2(output)

        output = self.fl(output)

        output = self.fc1(output)
        output = self.dropout1(output)

        output = self.fc2(output)
        output = self.dropout2(output)

        output = self.fc3(output)

        return output


main()
