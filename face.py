# Face Detection Program
# Author: Mba Uchenna

# Importing dependencies
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

m = 2  # slope
b = 1  # intercept

num_samples = 100  # number of data points
x = np.random.uniform(-10, 10, num_samples)
y = m * x + b

dataset = []
for i, xi in enumerate(x):
    dataset.append([x[i], y[i]])
#
# plt.scatter(x, y)
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.title('Generated Sample Linear Data')
# plt.show()

# principle components
cov = np.cov(dataset, rowvar=False)
eig_values, eig_vectors = np.linalg.eig(cov)

sort_indices = np.argsort(eig_values)[::-1]  # Sort in descending order
sorted_eigenvalues = eig_values[sort_indices]
sorted_eigenvectors = eig_vectors[:, sort_indices]

pc = np.dot(dataset, sorted_eigenvectors)

print(pc)
