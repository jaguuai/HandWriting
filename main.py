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
# img_index = 8888  # Update this value to look at other images
# if img_index >= len(X_train_display):
#     raise IndexError(f"img_index {img_index} is out of bounds for the dataset with length {len(X_train_display)}")

# img = X_train_display[img_index].numpy()
# label = y_train[img_index].item()
# print(f"Image label: {chr(label + 96)}")
# plt.imshow(img, cmap='gray')
# plt.show()


# Step 2.1 - Step 3.1: Modeli oluşturma ve eğitim
from sklearn.neural_network import MLPClassifier

mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=100, alpha=1e-4, solver="adam", verbose=10, tol=1e-4, random_state=1)
print("Created MLP network")

mlp1.fit(X_train_scaled, y_train)

# Print training and test set scores
print("Training set score: %f" % mlp1.score(X_train_scaled, y_train))
print("Test set score: %f" % mlp1.score(X_test_scaled, y_test))

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

# Step 3.3
# You can change this to any letters taht you think the neural network may have confused
predicted_letter="u"
actual_letter="v"

# This code counts all mistakes for the letters above
mistake_list=[]
for i in range(len(y_test)):
    if(y_test[i]==(ord(actual_letter)-96)and y_test[i]==(ord(predicted_letter)-96)):
        mistake_list.append(i)
print("There were"+str(len(mistake_list))+"times that the letter"+actual_letter
   +"was predicted to be the letter"+predicted_letter+".")
# Once we know how many mistakes were made,we can change this to see an image of it
mistake_to_show=4 # e.g., change this to 3 if you want to see the 4th mistake

# This code checks that the number mistake you asked for can be shown and if so,dispays an image of it
if(len(mistake_list)>mistake_to_show):
    img = X_test[mistake_list[mistake_to_show]]
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()
else:
    print("Couldn't show mistake number"+str(mistake_to_show+1)+"because there were only"+ str(len(mistake_list))+"mistakes to show!")
           
# Step 4.1  
path, dirs, files = next(os.walk("letters_mod"))
files.sort()

# Step 4.2  
import cv2
# This code processes all the scanned images and adds them to the handwritten_story
handwritten_story = []
for i in range(len(files)):
    img_path = os.path.join("letters_mod", files[i])
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        handwritten_story.append(img)
    else:
        print(f"Could not read image: {img_path}")

print("Imported the scanned images.")

# Check if handwritten_story has any images
if len(handwritten_story) > 4:
    plt.imshow(handwritten_story[4], cmap='gray')  # <--- Change this index to see different letters
else:
    print("Not enough images to display the fifth one.")

# Step 4.3   
import numpy as np
typed_story = ""

for letter in handwritten_story:
    letter_resized = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_CUBIC)
    single_item_array = np.array(letter_resized).reshape(1, 784)
    prediction = mlp1.predict(single_item_array)
    typed_story += chr(prediction[0] + 96)

print("Conversion to typed story complete!")
print(typed_story)

## STEP 4.3

# This is a library we need to do some math on the image to be able to give it to the MLP in the right format
import numpy as np

typed_story = ""
for letter in handwritten_story:
  letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_CUBIC)

  #this bit of code checks to see if the image is just a blank space by looking at the color of all the pixels summed
  total_pixel_value = 0
  for j in range(28):
    for k in range(28):
      total_pixel_value += letter[j,k]
  if total_pixel_value < 20:
    typed_story = typed_story + " "
  else:         #if it NOT a blank, it actually runs the prediction algorithm on it
    single_item_array = (np.array(letter)).reshape(1,784)
    prediction = mlp1.predict(single_item_array)
    typed_story = typed_story + str(chr(prediction[0]+96))

print("Conversion to typed story complete!")
print(typed_story)

# STEP 4.4

# These steps process the scanned images to be in the same format and have the same properties as the EMNIST images
# They are described by the EMNIST authors in detail here: https://arxiv.org/abs/1702.05373v1
processed_story = []

for img in handwritten_story:
  #step 1: Apply Gaussian blur filter
  img = cv2.GaussianBlur(img, (7,7), 0)

  #steps 2 and 3: Extract the Region of Interest in the image and center in square
  points = cv2.findNonZero(img)
  x, y, w, h = cv2.boundingRect(points)
  if (w > 0 and h > 0):
    if w > h:
      y = y - (w-h)//2
      img = img[y:y+w, x:x+w]
    else:
      x = x - (h-w)//2
      img = img[y:y+h, x:x+h]

  #step 4: Resize and resample to be 28 x 28 pixels
  img = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)

  #step 5: Normalize pixels and reshape before adding to the new story array
  img = img/255
  img = img.reshape((28,28))
  processed_story.append(img)

print("Processed the scanned images.")

import matplotlib.pyplot as plt
plt.imshow(processed_story[4]) #<<< change this index if you want to see a different letter from the story
     
# STEP 4.5

# This is a library we need to do some math on the image to be able to give it to the MLP in the right format


typed_story = ""
for letter in processed_story:
  #this bit of code checks to see if the image is just a blank space by looking at the color of all the pixels summed
  total_pixel_value = 0
  for j in range(28):
    for k in range(28):
      total_pixel_value += letter[j,k]
  if total_pixel_value < 20:
    typed_story = typed_story + " "
  else:         #if it NOT a blank, it actually runs the prediction algorithm on it
    single_item_array = (np.array(letter)).reshape(1,784)
    prediction = mlp1.predict(single_item_array)
    typed_story = typed_story + str(chr(prediction[0]+96))

print("Conversion to typed story complete!")
print(typed_story)



















 





































