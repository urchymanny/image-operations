# Face Detection Program
# Author: Mba Uchenna

# Importing dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Directory path containing the images
directory = "images/"

total_pixels = 250*250

# Read all images in the directory
faces = []
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(directory, filename)
        image = cv.cvtColor(cv.imread(image_path), cv.COLOR_RGB2GRAY)

        # Get the image dimensions
        height, width = image.shape

        # Calculate the total number of pixels
        total_pixels = height * width
        image = image.reshape(total_pixels,)

        if image is not None:
            faces.append(image)


faces = np.asarray(faces)
faces = faces.transpose()


# NORMALIZING THE FACE VECTORS
avg_faces = faces.mean(axis=1)
avg_faces = avg_faces.reshape(faces.shape[0], 1)
normalized_face = faces - avg_faces


# Covariance Matrix & Eigen Values and Vectors
covariance_m = np.cov(np.transpose(normalized_face))
eig_values, eig_vectors = np.linalg.eig(covariance_m)


# Sort the array
eig_vectors = np.sort(eig_vectors)

# k = amount of faces to be detected
k = 1
k_eigen_vectors = eig_vectors[0:k, :]

# converting lower eig vectors to original dimensions
eigen_faces = k_eigen_vectors.dot(normalized_face.T)
weights = (normalized_face.T).dot(eigen_faces.T)


"""
TESTING FACE DETECTOR
"""

test_img = cv.cvtColor(cv.imread('jack.jpg'), cv.COLOR_RGB2GRAY)
test_img = test_img.reshape(total_pixels,1)

test_normalized_face_vector = test_img - avg_faces

test_weight = (test_normalized_face_vector.T).dot(eigen_faces.T)

index = np.argmin(np.linalg.norm(test_weight - weights, axis=1))

print(index)


# Display the image using plt
# plt.imshow(test_img)
# plt.show()