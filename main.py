
import os
import requests
import shutil
import zipfile

# Define the dataset path
dataset_path = os.path.join(os.path.expanduser('~'), '.emnist')

# Check if the EMNIST dataset is already downloaded
if not os.path.exists(dataset_path):
    print("EMNIST dataset not found. Downloading...")
    # Create the dataset directory if it doesn't exist
    os.makedirs(dataset_path, exist_ok=True)
    
    # Download the dataset zip file
    url = "http://www.itl.nist.gov/iaui/vip/cs_links/EMNIST/matlab.zip"
    zip_path = os.path.join(dataset_path, "matlab.zip")
    
    # Try to download the file
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # Check if the request was successful
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(response.raw, f)
        print("EMNIST dataset downloaded successfully!")
    except Exception as e:
        print("Error downloading EMNIST dataset:", e)
        
    # Try to extract the zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        print("EMNIST dataset extracted successfully!")
    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip file.")
    except Exception as e:
        print("Error extracting the zip file:", e)
    
    # Remove the zip file
    os.remove(zip_path)

# Now try to extract samples
try:
    from emnist import extract_training_samples

    X, y = extract_training_samples('letters')

    # Make sure that every pixel in all of the images is a value between 0-1 
    X = X / 255

    # Use the first 60000 instances as training and the next 10000 as testing
    X_train, X_test = X[:60000], X[60000:70000]
    y_train, y_test = y[:60000], y[60000:70000]

    # Reshape images
    X_train = X_train.reshape(60000, 784)
    X_test = X_test.reshape(10000, 784)

    print("Extracted our samples and divided our training and testing data sets")

    import matplotlib.pyplot as plt

    img_index = 8888   # <<<<<You can update this value to look at other images
    img = X_train[img_index]
    print("Image label: " + str(chr(y_train[img_index] + 96)))
    plt.imshow(img.reshape((28, 28)))
    plt.show()

except Exception as e:
    print("Error:", e)
