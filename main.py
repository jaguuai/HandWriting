
from emnist import extract_training_samples




# X will be our images and y will be labels
X,y=extract_training_samples("letter")

# Make sure that every pixel in all of the images is a value between 0-1 
X=X/255

# Use the first 60000 instances as training  and the next  10000 as texting
X_train,X_test=X[:60000],X[60000:70000]
y_train,y_test=y[:60000],y[60000:70000]

# There is one other thing we need to do,
# record the number of samples in each dataset and number of pixel in each image
X_train=X_train.reshape(60000,784)
X_test=X_test.reshape(10000,784)
print("Extracted our samples and divided our training and testing data sets")

import matplotlib.pyplot as plt
 
img_index=8888   #<<<<<You can update this value to look at other images
img=X_train[img_index]
print("Image label: "+str(chr(y_train[img_index]+96)))
plt.imshow(img.reshape((28,28)))

