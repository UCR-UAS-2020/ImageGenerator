# Outlined in:
# http://opencvpython.blogspot.com/2012/06/fast-array-manipulation-in-numpy.html
# https://stackoverflow.com/questions/11064454/adobe-photoshop-style-posterization-and-opencv


import numpy as np
import cv2

image = cv2.imread('IMG_0602.JPG')

quantization_level = 5

# Get colors
colors = np.arange(0, 256)

# Create divider
divider = np.linspace(0, 255, quantization_level + 1)[1]

# Get the array of quantization colors
quantization_colors = np.int0(np.linspace(0, 255, quantization_level))

# Get the divided color levels
color_levels = np.clip(np.int0(colors / divider), 0, quantization_level - 1)

# Create the palette
palette = quantization_colors[color_levels]

# Apply the palette
posterized_image = palette[image]

# Convert back into regular image format
posterized_image = cv2.convertScaleAbs(posterized_image)

# Resize the window
cv2.namedWindow('Posterized Image', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Posterized Image', 800, 800)

# Show the image
cv2.imshow('Posterized Image',posterized_image)

cv2.waitKey(0)
cv2.destroyAllWindows()