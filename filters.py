import cv2 as cv
import numpy as np

# Open the image.
img = cv.imread("mandriill.png")

# filters
darken = img - 128
low_contrast = (img / 2)
nl_low_contrast = ((img/255)**(1/3))*255
invert = 255 - img
lighten = img + 128
raised_contrast = img + 2
nl_raised_contrast = ((img/255)**(2))*255

print("Original - X")
cv.imshow(img)

print("Darkened")
cv.imshow(darken)

print("Low Contrast")
cv.imshow(low_contrast)

print("Non Linear Low Contrast")
cv.imshow(nl_low_contrast)

print("Inverted")
cv.imshow(invert)

print("Lightened")
cv.imshow(lighten)

print("Raised Contrast")
cv.imshow(raised_contrast)

print("Non Linear Raised Contrast")
cv.imshow(nl_raised_contrast)

# defining the transformation function
def transformation_function(input_pixel, r1, s1,  r2, s2):
  if 0 <= input_pixel <= r1:
    output_pixel = (s1/r1)*input_pixel
  elif r1 <= input_pixel <= r2:
    output_pixel = (s2-s1/r2-r1)*(input_pixel-r1)+s1
  else:
    output_pixel = (255-s2/255-r2)*(input_pixel-r2)+s2
  return output_pixel

# attempting to apply the transformation function to each pixel
output_image = np.array(img)

for i, dimension in enumerate(img):
  for arr_id, arr in enumerate(dimension):
    for p_id, point in enumerate(arr):
      input_pixel = img[i][arr_id][p_id]
      output_image[i][arr_id][p_id] = transformation_function(input_pixel,0.1,0.2,0.4,0.5)

print("Image transformed and processed!")
cv.imshow(output_image)

cv.imshow('custom image', img)
cv.waitKey(0)