# Step 1.1 - Step 1.3: Veri yükleme ve hazırlama
import os
import torch
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
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

# Reshape images for sklearn MLP
X_train = train_dataset.data.view(train_length, -1)
y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.view(test_length, -1)
y_test = test_dataset.targets.numpy()

# Reshape images for display
X_train_display = X_train.view(-1, 28, 28)
X_test_display = X_test.view(-1, 28, 28)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 1.3
# Display a sample image
img_index = 8888  # Update this value to look at other images
if img_index >= len(X_train_display):
    raise IndexError(f"img_index {img_index} is out of bounds for the dataset with length {len(X_train_display)}")

img = X_train_display[img_index].numpy()
label = y_train[img_index].item()
print(f"Image label: {chr(label + 96)}")
plt.imshow(img, cmap='gray')
plt.show()


# # Step 2.1 - Step 3.1: Modeli oluşturma ve eğitim
# from sklearn.neural_network import MLPClassifier

# mlpl = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4, solver="adam", verbose=10, tol=1e-4, random_state=1)
# print("Created MLP network")

# mlpl.fit(X_train_scaled, y_train)

# # Print training and test set scores
# print("Training set score: %f" % mlpl.score(X_train_scaled, y_train))
# print("Test set score: %f" % mlpl.score(X_test_scaled, y_test))

# # Step 3.2: Confusion matrix
# from sklearn.metrics import confusion_matrix


# y_pred = mlpl.predict(X_test_scaled)
# cm = confusion_matrix(y_test, y_pred)

# plt.matshow(cm)
# plt.title('Confusion matrix')
# plt.colorbar()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()
