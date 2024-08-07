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
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    transformer=transforms.Compose([
        transforms.Resize((200,200)), # Resize image to 200x200 pixels
        transforms.ToTensor(),  # Convert image/Numpy Array to PyTorch tensor and scale pixel values to [0, 1]
        transforms.Normalize([0.5, 0.5, 0.5],  # Normalize tensor to have mean 0 and std 1 (scales to [-1, 1]),formula (x-mean)/std
                             [0.5, 0.5, 0.5]),
    ])
    training_path='E:/Shaheer Data/Comsats/Semester7/machineLearning/Assignment2/Data/train'
    testing_path='E:/Shaheer Data/Comsats/Semester7/machineLearning/Assignment2/Data/test'
    # Create a DataLoader for the training data
    # - Load images from the specified directory
    # - Apply the defined transformations (resize, convert to tensor, normalize)
    # - Organize images into batches of 32
    # - Shuffle the data at the beginning of each epoch
    train_data= torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(training_path,transform=transformer),
        batch_size=32,shuffle=True
    )
    test_data = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(testing_path,transform=transformer),
        batch_size=32, shuffle=True
    )
    # Initialize the ConvNet model with 10 output classes
    model =ConvNet(10)
    # Set up the Adam optimizer with a learning rate of 0.1 for model parameters
    optimizer = Adam(model.parameters(), lr=0.0005)
    # Define the loss function as cross-entropy loss for classification tasks
    loss_func = nn.CrossEntropyLoss()
    # Set the number of epochs for training the data
    num_epochs = 10
    # Create a array of epoch indices from 0 to num_epochs-1 for tracking or plotting training progress.
    epoch_list = [i for i in range(num_epochs)]

    # Store the Taining Accuracy and loss at each epoch
    train_accuracy_list = []
    loss_accuracy_list = []

    # Calculate the number of training images by counting all .jpg files in the specified directory and its subdirectories
    #glob.glob() is a function that returns a list of file paths matching a specified pattern
    #** matches directories at any level, including subdirectories of subdirectories. 'folder/**/*.jpg' matches all .jpg files in the folder directory and any of its subdirectories, regardless of how deeply nested they are.
    train_count = len(glob.glob(training_path + '/**/*.jpg'))
    test_count = len(glob.glob(testing_path + '/**/*.jpg'))

    for epoch in range(num_epochs):
        # Set the model to training mode to enable features like dropout and batch normalization
        model.train()
        # Initialize the variable to accumulate the total training loss for the epoch
        train_loss = 0
        # Initialize the variable to accumulate the total training accuracy for the epoch
        train_accuracy = 0
        total_samples =0
        # Iterate over the training data batches, where 'i' is the batch index, and 'images' and 'label' are the current batch of input images and their corresponding labels/output
        for i,(images,label) in enumerate(train_data):
            #images[0] contain all 32 batch of image and label contain all 32 label correspond to each images

            # Display the first image of the current batch using Matplotlib with a grayscale colormap,
            # and set the plot title to show the label of the first image in the batch
            # plt.imshow(images[0][0], cmap='gray')  # Show the image
            # plt.title(f"Label: {label[0]}")  # Set the plot title to the image's label
            # plt.show()  # Display the plot

            # Perform a forward pass through the model to get predictions for the input images yhat will contain the model's predicted outputs for the input images.
            yhat = model(images)
            # Compute the loss between the model's predictions yhat and the true labels using the loss function measures how well the model’s predictions match the actual labels
            loss = loss_func(yhat, label)

            # apply backpropgation
            # Clear previous gradients to prevent accumulation(from affecting the current batch’s update) from previous batches
            optimizer.zero_grad()
            # Compute gradients of the loss with respect to model parameters using backpropagation
            loss.backward()
            # Update model parameters using the optimizer based on computed gradients by loss.backward()
            optimizer.step()
            # Add the loss value for the current batch to the total training loss for one epoch
            train_loss += loss.item()
            # Get the predicted class indices by finding the index with the maximum value in yhat (predictions)
            _, prediction = torch.max(yhat.data, 1)
            # Count the number of correct predictions in the current batch and accumulate it to the total training accuracy for one epoch
            train_accuracy += int(torch.sum(prediction == label.data))
        # Compute the average training accuracy after epoch by dividing the total correct predictions to total Data images
        train_accuracy = train_accuracy / train_count
        # Compute the average training loss after the epoch by dividing the total loss by the total Data Images
        train_loss = train_loss / train_count

        #     _,prediction=torch.max(yhat,1)
        #     total_samples += label.size(0)
        #     train_accuracy+=(prediction == label).sum().item()
        # train_accuracy = train_accuracy / total_samples
        # train_loss = train_loss / total_samples

        train_accuracy_list.append(train_accuracy)
        loss_accuracy_list.append(train_loss)
        print(f'Epoch :{epoch} loss :  {train_loss} accuracy : {train_accuracy}')

    # Save the model's state dictionary (parameters) to a file named 'model.pt'
    # with open('model.pt', 'wb') as f:
    #     torch.save(model.state_dict(), f)
    print("Model Saved")
    torch.save(model,'E:/Shaheer Data/Comsats/Semester7/machineLearning/Assignment2/model2.pth')
    plt.plot(epoch_list, train_accuracy_list, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(epoch_list, loss_accuracy_list, label='Train error')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


class ConvNet(nn.Module):
    def __init__(self, numofclasses):
        super(ConvNet,self).__init__()

        # Input shape= (32,3,200,200)
        # Define the first convolutional layer with 3 input channels RGB image and 12 output channels
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
        # Apply batch normalization to the output of the previous layer with 35 feature channels.
        # This helps to stabilize and accelerate training by normalizing the activations of the layer
        self.bn1 = nn.BatchNorm2d(num_features=12)
        self.ReLU1=nn.ReLU() # Apply ReLU activation

        # Shape= (32,12,200,200)
        # Define the second convolutional layer with 12 input channels and 25 output channels
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=25, kernel_size=3, stride=1, padding=1)
        self.ReLU2 = nn.ReLU() # Apply ReLU activation

        # Shape= (32,25,200,200)
        # Define max pooling layer with 2x2 kernel and stride of 2
        self.pool1=nn.MaxPool2d(kernel_size=2,stride=2)

        # Shape= (32,25,100,100)
        self.conv3 = nn.Conv2d(in_channels=25, out_channels=35, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=35)
        self.ReLU3 = nn.ReLU() # Apply ReLU activation

        # Shape= (32,35,100,100)
        self.conv4 = nn.Conv2d(in_channels=35, out_channels=15, kernel_size=3, stride=1, padding=1)
        self.ReLU4 = nn.ReLU() # Apply ReLU activation

        # Shape= (32,15,100,100)
        # Define max pooling layer with 2x2 kernel and stride of 2
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Shape= (32,15,50,50)
        # Flatten layer to convert 2D feature maps to 1D tensor
        self.fl=nn.Flatten()

        # Define the first fully connected layer with input features and 200 output features
        self.fc1=nn.Linear(in_features=50*50*15,out_features=500)
        self.dropout1 = nn.Dropout(0.5)

        # Define the second fully connected layer with 200 input features and output features equal to numofclasses=10
        self.fc2 = nn.Linear(in_features=500, out_features=100)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3=nn.Linear(in_features=100,out_features=numofclasses);

    def forward(self,input):
        output=self.conv1(input)
        output=self.bn1(output)
        output = self.ReLU1(output)
         
        output=self.conv2(output)
        output = self.ReLU2(output)
         
        output = self.pool1(output)
         
        output = self.conv3(output)
        output=self.bn2(output)
        output = self.ReLU3(output)
         
        output = self.conv4(output)
        output = self.ReLU4(output)
         
        output = self.pool2(output)
         
        output=self.fl(output)
         
        output = self.fc1(output)
        output = self.dropout1(output)

        output = self.fc2(output)
        output = self.dropout2(output)

        output = self.fc3(output)

        return output

main()
