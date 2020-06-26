# Adapted from these sources:
# https://www.learnopencv.com/alpha-blending-using-opencv-cpp-python/
#
import numpy as np
import cv2
import random

from typing import List

import json
# from ImageGenerator.proto import *
from proto import *
# from ImageGenerator.target_gen import create_target_image
from target_gen import create_target_image
# TODO: re-implement above imports
# TODO: Generate this target an put it in 0602.jpg:


# Test placeholder target object. we will eventually randomize our own
# target1 = Target(alphanumeric='n',
#                  shape=Shape.Triangle,
#                  alphanumeric_color=Color.White,
#                  shape_color=Color.Blue,
#                  posx=100,
#                  posy=100,
#                  scale=18
#                  )


# pseudo code:
# iH, iW = image height, image width

# return a single random target by picking from the enums in Proto
def make_random_target(image_height, image_width):
    alphanum = random.choice(list(Alphanum))
    shape = random.choice(list(Shape))
    alphanum_color = random.choice(list(Color))
    shape_color = random.choice(list(Color))
    scale = random.randint(10, 20)
    width = int(image_width * (scale / 100.))
    height = int(image_height * (scale / 100.))
    while alphanum_color == shape_color:
        shape_color = random.choice(list(Color))
    x = random.randint(width, int(image_width-width*1.5))
    y = random.randint(height, int(image_height-height*1.5))
    rotation = random.randint(0, 359)

    return Target(alphanumeric=alphanum,
                  shape=shape,
                  alphanumeric_color=alphanum_color,
                  shape_color=shape_color,
                  posx=x,
                  posy=y,
                  scale=scale,
                  rotation=rotation
                  )


# return a list of targets by calling make_random_target() several times
# choose a random number of targets n = [0, 5]
def make_random_target_list(image_height, image_width):
    target_list = []
    for i in range(0, 2):
        target_list.append(make_random_target(image_height, image_width))
    return target_list


# Creates a cv2 representation of a image with the superimposed random targets
def make_image(t_list, im_input):
    im_output = im_input / 255.
    for target in t_list:
        im_output = push_target_to_im(im_output * 255., target)
    return im_output


def make_target_dict_json(t_list):
    targ_out_dict = {}
    i = 1
    for targ in t_list:
        # can index from 0 or 1
        targ_out_dict[i] = targ.make_json()
        i += 1

        # targ_out_dict[targ.key] = targ.make_json(1)
    # for index, targ in enumerate(t_dict):
    #     targ_out_dict.update({str(index), targ.make_json()})
    print(json.dumps(targ_out_dict, indent=2))
    return json.dumps(targ_out_dict, indent=2)


def write_image_crop(filename: str, image, target: Target):
    # https://www.geeksforgeeks.org/python-opencv-cv2-imwrite-method/
    # TODO: refactor image cropping based on target.scale
    # get the target position and scale
    scale = target.scale
    x = target.x
    y = target.y
    # calculate a bounding box by taking position.x +- size and position .y +- size
    # (x, y)
    top_left = (x - scale, y - scale)
    # (x, y)
    bot_right = (x + scale, y + scale)

    # slice out subarray
    img = image[top_left[1]:bot_right[1]+1, top_left[0]:bot_right[0]+1]

    # save to file
    cv2.imwrite(filename, img)



# t_dict = make_random_target_list()
# im_out = make_image()



# This function should return an cv2 image:
# Input: target_dict is a dictionary of several subdictionaries that are in the form like target1 above
# def create_target_image(target_dict)


rotation = 0
position = (100, 100)
scale_percent = 20  # percent of original size

# Parameters for input shape
# img_shape = cv2.imread()


# Read the images
# img_target = create_target_image_test()


def push_target_to_im(im: np.ndarray, target: Target) -> np.ndarray:
    scale_percent = target.scale
    rotation = target.rotation
    x = target.x
    y = target.y

    img_target = create_target_image(target)

    width = int(img_target.shape[1] * (scale_percent / 100.))
    height = int(img_target.shape[0] * (scale_percent / 100.))
    dim = (width, height)
    # resize image
    img_target = cv2.resize(img_target, dim, interpolation=cv2.INTER_AREA)
    # img_background = cv2.imread(r'.\IMG_0602.JPG')
    img_background = im
    # add a 255-alpha channel to background
    if img_background.shape[2] == 4:
        img_background = img_background[:, :, 0:3]

    img_background = np.dstack((img_background, 255. * np.ones(np.shape(img_background)[0:2])))
    rows, cols, depth = img_target.shape

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation, 1)
    img_target = cv2.warpAffine(img_target, M, (cols, rows))

    letter_size = np.shape(img_target)
    background_size = np.shape(img_background)

    # add the letter image to a blank image which is the size of the background
    img_letter1 = np.zeros(np.shape(img_background))
    target_height = img_target.shape[0]
    target_width = img_target.shape[1]
    img_letter1[y:y + target_height, x:x + target_width, :] = \
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



    return outImage



# Scale transform
'''
width = int(img_target.shape[1] * scale_percent / 100)
height = int(img_target.shape[0] * scale_percent / 100)
dim = (width, height)
# resize image
img_target = cv2.resize(img_target, dim, interpolation=cv2.INTER_AREA)

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
'''
if __name__ == '__main__':
    # img = cv2.imread('background-of-green-and-healthy-grass-royalty-free-image-1586800097.jpg')
    img = cv2.imread('background-of-green-and-healthy-grass-royalty-free-image-1586800097.jpg')
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    t_list = make_random_target_list(resized.shape[1], resized.shape[0])
    new_img = make_image(t_list, resized)
    json = make_target_dict_json(t_list)
    cv2.imshow('win', new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('1.png', new_img * 255.)
    file = open(r'1.json', 'a')
    file.write(json)
    file.close()




