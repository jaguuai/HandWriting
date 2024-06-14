# Step 1.1
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt

# Define the dataset path
dataset_path = os.path.join(os.path.expanduser('~'), '.emnist')

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Download and load the training data
train_dataset = datasets.EMNIST(root=dataset_path, split='letters', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Download and load the test data
test_dataset = datasets.EMNIST(root=dataset_path, split='letters', train=False, transform=transform, download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

print("EMNIST dataset downloaded and loaded successfully!")
# Step 1.2
# Get the lengths of the datasets
train_length = len(train_dataset)
test_length = len(test_dataset)

print(f"Number of training images: {train_length}")
print(f"Number of test images: {test_length}")

# To find the maximum index for training images
print(f"Maximum index for training images: {train_length - 1}")

# To find the maximum index for test images
print(f"Maximum index for test images: {test_length - 1}")

# Extract all training samples
X_train = torch.zeros((train_length, 1, 28, 28))
y_train = torch.zeros(train_length, dtype=torch.long)
start_idx = 0

for batch_idx, (data, target) in enumerate(train_loader):
    batch_size = data.size(0)
    end_idx = start_idx + batch_size
    X_train[start_idx:end_idx] = data
    y_train[start_idx:end_idx] = target
    start_idx = end_idx

X_test = torch.zeros((test_length, 1, 28, 28))
y_test = torch.zeros(test_length, dtype=torch.long)
start_idx = 0

for batch_idx, (data, target) in enumerate(test_loader):
    batch_size = data.size(0)
    end_idx = start_idx + batch_size
    X_test[start_idx:end_idx] = data
    y_test[start_idx:end_idx] = target
    start_idx = end_idx
    
# Step 1.3    
# Display a sample image
img_index = 1000  # Update this value to look at other images
if img_index >= len(X_train):
    raise IndexError(f"img_index {img_index} is out of bounds for the dataset with length {len(X_train)}")

img = X_train[img_index].numpy().squeeze()
label = y_train[img_index].item()
print(f"Image label: {chr(label + 96)}")
plt.imshow(img, cmap='gray')
plt.show()

# Reshape images for sklearn MLP
X_train = X_train.view(train_length, -1)
X_test = X_test.view(test_length, -1)

# Convert torch tensors to numpy arrays
X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

# Import ML libraries
from sklearn.neural_network import MLPClassifier

# Create and train the MLP model
mlpl = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=4, solver="sgd", verbose=10, tol=1e-4, random_state=1, learning_rate_init=1)
print("Created first MLP network")

# Train the model
mlpl.fit(X_train, y_train)

# Print training and test set scores
print("Training set score: %f" % mlpl.score(X_train, y_train))
print("Test set score: %f" % mlpl.score(X_test, y_test))

