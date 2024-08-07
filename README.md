<h1>DryFruits_Image_Classifier_CNN_Project</h1>

<h2>Overview</h2>
<p>This project involves training and testing a convolutional neural network (CNN) on an image dataset. The model is implemented using PyTorch and includes both training and testing scripts.</p>

<h2>Dataset</h2>
    <p>The dataset used for this project is the Tree Nuts Image Classification dataset, available on Kaggle. You can download it from the following link:</p>
    <p><a href="https://www.kaggle.com/datasets/gpiosenka/tree-nuts-image-classification">Tree Nuts Image Classification Dataset</a></p>
    <p>All data related to the nuts images for this project is stored in a file named <code>Data</code>.</p>

<h2>Training Script</h2>
<p>The training script is designed to train a CNN model on a set of images. Below are the key components of the script:</p>

<h3>Setup</h3>
<ul>
    <li>The script checks if a GPU is available and sets the device accordingly.</li>
    <li>Image transformations are applied, including resizing, converting to tensor, and normalizing.</li>
</ul>

<h3>Data Loading</h3>
<ul>
    <li>Training and testing datasets are loaded from specified directories using <code>torchvision.datasets.ImageFolder</code>.</li>
    <li>Paths to the training and testing data are set using the <code>training_path</code> and <code>testing_path</code> variables. Please ensure these paths are updated to point to your own dataset directories.</li>
</ul>

<h3>Model</h3>
<ul>
    <li>A custom CNN model <code>ConvNet</code> is defined with several convolutional layers, batch normalization, ReLU activations, pooling layers, and fully connected layers.</li>
</ul>

<h3>Training Process</h3>
<ul>
    <li>The model is trained for a specified number of epochs. During each epoch, the model processes batches of images, computes loss, performs backpropagation, and updates model weights.</li>
    <li>Training loss and accuracy are computed and plotted at the end of each epoch.</li>
    <li>The trained model is saved to a file named <code>model1.pth</code>.</li>
</ul>

<h2>Testing Script</h2>
<p>The testing script evaluates the performance of the trained model on a test dataset. Here are the main components:</p>

<h3>Setup</h3>
<ul>
    <li>Similar to the training script, it checks for GPU availability and sets the device.</li>
    <li>Image transformations are applied to the test data in the same way as for training.</li>
</ul>

<h3>Data Loading</h3>
<ul>
    <li>The test dataset is loaded from a specified directory using <code>torchvision.datasets.ImageFolder</code>. Update the <code>testing_path</code> variable to point to your test dataset directory.</li>
</ul>

<h3>Model Evaluation</h3>
<ul>
    <li>The trained model is loaded from the saved file <code>model1.pth</code>.</li>
    <li>The script calculates the accuracy of the model on the test dataset and prints the results.</li>
</ul>

<h2>Important Notes</h2>
<ul>
    <li>Make sure to update the <code>training_path</code> and <code>testing_path</code> variables in both scripts to point to the correct directories where your data is stored.</li>
    <li>Ensure that the saved model file <code>model1.pth</code> is correctly specified in the testing script.</li>
</ul>

<h2>Dependencies</h2>
<ul>
    <li>PyTorch</li>
    <li>torchvision</li>
    <li>matplotlib</li>
</ul>

<ul>
    <li>To run the project, follow these steps:</li>
    <li><strong>Create a Virtual Environment:</strong>
        <ul>
            <li>Open your  command prompt , navigate to your project directory and Create a virtual environment with:
                <pre><code>python -m venv env</code></pre>
            </li>
            <li>Activate the virtual environment:
                <pre><code>.\env\Scripts\activate</code></pre>
            </li>
        </ul>
    </li>
    <li><strong>Install Dependencies:</strong>
        <ul>
            <li>With the virtual environment activated, install the required dependencies:
                <pre><code>pip install torch torchvision matplotlib</code></pre>
            </li>
        </ul>
    </li>
    <li><strong>Run the Project:</strong>
        <ul>
            <li>After installing the dependencies, you can run the training and testing scripts as needed. To run the project You write command :</li>
            <pre><code>python ProjectName.py</code></pre>
        </ul>
    </li>
</ul>
