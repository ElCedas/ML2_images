import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import squeezenet1_1
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from torchvision import datasets, transforms
import seaborn as sns

class CNN(nn.Module):
    """Convolutional Neural Network model for image classification using SqueezeNet."""
    
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        """CNN model initializer."""
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Freeze all parameters initially
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze the last few layers as specified
        if unfreezed_layers > 0:
            for child in list(self.base_model.children())[-unfreezed_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

        # Replace the classifier of the SqueezeNet model
        # SqueezeNet uses 'classifier' as its final part rather than 'fc'
        self.base_model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        self.base_model.num_classes = num_classes
        
    def forward(self, x):
        """Forward pass of the model."""
        x = self.base_model(x)
        return x
    def predict(self, image):
        return(self(image))


class CNN2(nn.Module):
    """Convolutional Neural Network model for image classification."""
    
    def __init__(self, base_model, num_classes, unfreezed_layers=0):
        """CNN model initializer."""
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes

        # Freeze convolutional layers
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze specified number of layers
        if unfreezed_layers > 0:
            children = list(self.base_model.children())
            for child in children[-unfreezed_layers:]:
                for param in child.parameters():
                    param.requires_grad = True

        # Modify the classifier to match the number of classes
        in_features = self.base_model.classifier[-1].in_features
        self.base_model.classifier[-1] = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        """Forward pass of the model."""
        x = self.base_model(x)
        return x

    def train_model(self, train_loader, valid_loader, optimizer, criterion, epochs, nepochs_to_save=10, start_epoch=0):
        """Train the model, save checkpoints, and output training and validation metrics."""
        for epoch in range(epochs):
            self.train()
            train_loss = train_accuracy = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_accuracy += (outputs.argmax(1) == labels).sum().item()

            train_loss /= len(train_loader)
            train_accuracy /= len(train_loader.dataset)

            # Evaluate on validation set
            self.eval()
            valid_loss = valid_accuracy = 0.0
            with torch.no_grad():
                for images, labels in valid_loader:
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    valid_loss += loss.item()
                    valid_accuracy += (outputs.argmax(1) == labels).sum().item()

            valid_loss /= len(valid_loader)
            valid_accuracy /= len(valid_loader.dataset)

            print(f'Epoch {start_epoch + epoch + 1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_accuracy:.4f}')
            
            if (epoch + 1) % nepochs_to_save == 0:
                checkpoint_path = f'./models/cnn_checkpoint_epoch_{start_epoch + epoch + 1}.pt'
                torch.save(self.state_dict(), checkpoint_path)
                print(f'Saved checkpoint to {checkpoint_path}')


def load_model(checkpoint_path, num_classes):
    """Load the saved model from a checkpoint."""
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize the base model with SqueezeNet, make sure to match the architecture
    base_model = squeezenet1_1(pretrained=False)  # Load without pretrained weights
    model = CNN(base_model, num_classes, unfreezed_layers=2)  # Ensure CNN class is adapted for SqueezeNet if necessary
    
    # Load the saved state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

def evaluate_model(model, data_loader):
    """Evaluate the model accuracy on a provided data loader."""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# Example usage
category_names = [
    'Bedroom', 'Coast', 'Forest', 'Highway', 'Industrial',
    'Inside city', 'Kitchen', 'Living room', 'Mountain', 'Office',
    'Open Country', 'Store', 'Street', 'Suburb', 'Tall building'
]
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224
    transforms.ToTensor()  # Convert the image to a PyTorch tensor
])

valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),          # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize the image
                         std=[0.229, 0.224, 0.225])   # Using ImageNet mean and std
])

def load_model_mobile(checkpoint_path, num_classes):
    """ Load the saved model from a checkpoint. """
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Reinitialize the model with the correct number of classes
    base_model = torchvision.models.mobilenet_v3_large(weights=None)  # Load without pretrained weights
    model = CNN2(base_model, num_classes, unfreezed_layers=2)

    # Load the saved state dictionary into the model
    model.load_state_dict(checkpoint['model_state_dict'])

    return model
    
def test_CNN(img):
    """Process the image and return the prediction."""
    num_classes = 15  # Assuming 15 classes
    checkpoint_path = './Modelos/cnn_squeeze_final_epoch_50.pt'

    # Load the model
    model = load_model(checkpoint_path, num_classes)

    # Apply transformations
    img_tensor = transform(img).unsqueeze(0)  # Transform and add batch dimension

    # Perform prediction
    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction, dim=1)

    return category_names[predicted_class.item()]

def test(img):
    """Process the image and return the prediction."""
    num_classes = 15  # Make sure this matches the number of classes the model was trained on
    checkpoint_path = './Modelos/cnn_squeeze_final_epoch_50.pt'

    # Load the model
    model = load_model(checkpoint_path, num_classes)

    # Convert PIL Image to Tensor
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Perform prediction
    with torch.no_grad():
        prediction = model(img_tensor)
        predicted_class = torch.argmax(prediction, dim=1)

    return category_names[predicted_class.item()] # Return the predicted class index as an integer

    
def process_image_shape(img):
    """Process the image and return the shape."""
    # Convert image to numpy array
    image_array = np.array(img)
    return image_array.shape

def process_image_brightness(img):
    """Process the image and return the average brightness."""
    image_array = np.array(img.convert('L'))  # Convert image to grayscale
    return np.mean(image_array)

# Set up the Streamlit app
st.title('ML2, trabajo clasificacion Hugo y Alfonso')

# Create a dropdown menu for choosing the processing function
model_option = st.selectbox(
    "Choose a processing model:",
    ('mobilenet_v3_large','cnn_squeeze'),
    index=0  # Default to first option
)

# Create a file uploader to accept image uploads
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file)

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Process the image based on selected model
    if model_option == 'mobilenet_v3_large':
        result = test_CNN(image)
    elif model_option == 'Brightness of Image':
        result = process_image_brightness(image)
    elif model_option == 'cnn_squeeze':
        result = test(image)

    # Display the result of the processing
    st.write('Result of processing the image:')
    st.write(result)