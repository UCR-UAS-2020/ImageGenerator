# Adapted from these sources:
# https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
#

import numpy as np
import cv2

from ImageGenerator.proto import *
from ImageGenerator.target_gen import *

# TODO: Generate this target an put it in 0602.jpg:
target1 = Target(alphanumeric='n',
                 shape=Shape.Triangle,
                 alphanumeric_color=Color.White,
                 shape_color=Color.Blue,
                 pos=(100, 100),
                 scale=1
                 )

# This function should return an cv2 image:
# create_target_image(target1)


rotation = 0
position = (100, 100)
scale_percent = 20  # percent of original size

# Parameters for input shape
# img_shape = cv2.imread()


# Read the images
img_target = create_target_image_test()


# Scale transform

width = int(img_target.shape[1] * scale_percent / 100)
height = int(img_target.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img_target = cv2.resize(img_target, dim, interpolation=cv2.INTER_AREA)

# img_background = cv2.imread(r'C:\Users\chris\Desktop\UCRUAS-2020-Dragonfly-VIEW\image_gen\IMG_0602.JPG')
img_background = cv2.imread(r'.\IMG_0602.JPG')
# add a 255-alpha channel to background
img_background = np.dstack((img_background, 255. * np.ones(np.shape(img_background)[0:2])))

rows, cols, depth = img_target.shape

M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
img_target = cv2.warpAffine(img_target, M, (cols, rows))

letter_size = np.shape(img_target)
background_size = np.shape(img_background)

# add the letter image to a blank image which is the size of the background
img_letter1 = np.zeros(np.shape(img_background))
img_letter1[position[0]:position[0] + letter_size[0], position[1]:position[1] + letter_size[1], :letter_size[2]] = \
    img_target[:, :, :]

img_filter = img_letter1[:, :, 3]
# add an alpha channel by extending the same array to alpha
img_filter = np.repeat(img_filter[:, :, np.newaxis], 4, axis=2)


foreground = img_letter1
background = img_background
alpha = img_filter

# Convert u-int8 to float
foreground = foreground.astype(float)
background = background.astype(float) / 255.

# Normalize the alpha mask to keep intensity between 0 and 1
# alpha = alpha.astype(float) / 255

# Multiply the foreground with the alpha matte
foreground = cv2.multiply(alpha, foreground)

# Multiply the background with ( 1 - alpha )
background = cv2.multiply(1.0 - alpha, background)

# Add the masked foreground and background.
outImage = cv2.add(foreground, background)

# Display image
cv2.imshow("outImg", outImage)
cv2.waitKey(0)

cv2.imwrite('test.png', outImage*255.)
