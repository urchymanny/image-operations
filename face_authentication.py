import os
import cv2 as cv
import numpy as np

# Directory path containing training images
training_directory = "images/"

# Load training data from the directory
training_data = []
labels = []

for filename in os.listdir(training_directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(training_directory, filename)
        image = cv.imread(image_path)

        # Convert image to grayscale
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

        # Resize the image if needed
        # gray = cv.resize(gray, (desired_width, desired_height))

        # Flatten the image into a 1D vector
        flattened_image = gray.flatten()

        # Add the flattened image to the training data array
        training_data.append(flattened_image)

        # Add the label for the image (e.g., person's name or ID)
        labels.append(filename.split('.')[0])  # Assumes the file name is the label

# Convert training data and labels to numpy arrays
training_data = np.array(training_data)
labels = np.array(labels)

s = []
# Perform face authentication on a testing data image
def authenticate_face(test_image_path):
    # Load the testing data image
    test_image = cv.imread(test_image_path)

    # Convert the testing image to grayscale
    gray_test_image = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)

    # Flatten the testing image into a 1D vector
    flattened_test_image = gray_test_image.flatten()

    # Perform authentication by comparing the testing image with the training data
    for i, train_image in enumerate(training_data):
        similarity_score = np.dot(flattened_test_image, train_image) / (np.linalg.norm(flattened_test_image) * np.linalg.norm(train_image))
        s.append("{:.8f}".format(float(similarity_score)))
        # print(labels[i], ":", "{:.8f}".format(float(similarity_score)))
        if similarity_score > threshold:
            # Authentication successful, return the corresponding label
            return labels[i]


    # Authentication failed
    return None


# Set the threshold for face authentication
# might need an ai to determine the best threshold value
threshold = .00000057  # Adjust the value based on your requirements


# Test the authentication function with a testing image
test_image_path = "images/George_W_Bush_0004.jpg"  # Replace with the path to your testing image
result = authenticate_face(test_image_path)


# still trying to figure out the best value for the threshold
# print(np.sort(s))


if result is not None:
    print("Authentication successful. Face belongs to:", result)
else:
    print("Authentication failed. Face does not match any enrolled data.")
