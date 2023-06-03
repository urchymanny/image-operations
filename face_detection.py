# Face Detection Program
# Author: Mba Uchenna

# Importing dependencies
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Directory path containing the images
total_pixels = 250*250


def training_images(directory):
    images = []
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory, filename)
            image = cv.cvtColor(cv.imread(image_path), cv.COLOR_RGB2GRAY)
            image = image.reshape(total_pixels, )

            if image is not None:
                images.append(image)
            images.append(image)
    return images


def training(training_data):
    faces = np.asarray(training_data)
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
    k = 20
    k_eigen_vectors = eig_vectors[0:k, :]

    # converting lower eig vectors to original dimensions
    eigen_faces = k_eigen_vectors.dot(normalized_face.T)
    weights = (normalized_face.T).dot(eigen_faces.T)

    return avg_faces, eigen_faces, weights


def testing(test_img, avg_faces, eigen_faces, training_weights,):
    test_img = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img = test_img.reshape(total_pixels, 1)

    test_normalized_face_vector = test_img - avg_faces

    test_weight = (test_normalized_face_vector.T).dot(eigen_faces.T)

    index = np.argmin(np.linalg.norm(test_weight - training_weights, axis=1))

    return index


dir = "images/"
data = training_images(dir)

test_image = cv.imread('jack.jpg')
average_faces, eigen_f, t_weights = training(data)

print(testing(test_image, average_faces, eigen_f, t_weights))



# Display the image using plt
# plt.imshow(test_img)
# plt.show()