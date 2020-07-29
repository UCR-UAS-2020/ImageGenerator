import cv2
import numpy as np
import random

from json import dumps
from proto import *
from target_gen import create_target_image
from image_generator import make_random_target
from argument_parser import parse_target
# TODO: Fix outer border for some shape-rotation combination; some of the longer shapes are getting corners cut off


def write_target_to_im(target: Target, im: np.ndarray):
    img = create_target_image(target)

    width = int(img.shape[1] * (target.scale / 100.))
    height = int(img.shape[0] * (target.scale / 100.))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

    im_background = np.dstack((im, 255. * np.ones(np.shape(im)[0:2])))
    im_background = im_background[target.y:target.y+img.shape[0], target.x:target.x+img.shape[1]]
    rows, cols, depth = img.shape

    matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), target.rotation, 1)
    img = cv2.warpAffine(img, matrix, (cols, rows))

    img_letter = img

    img_filter = img_letter[:, :, 3]
    img_filter = np.repeat(img_filter[:, :, np.newaxis], 4, axis=2)

    foreground = img_letter.astype(float)
    background = im_background.astype(float) / 255.
    alpha = img_filter

    foreground = cv2.multiply(alpha, foreground)
    background = cv2.multiply(1.0 - alpha, background)

    return cv2.add(foreground, background)


def write_target_image_and_json(file_number: str, image: np.ndarray):
    target = make_random_target(image.shape[0], image.shape[1])
    target_img = write_target_to_im(target, image)
    json_string = target.make_target_only_json()
    cv2.imwrite('Targets/' + file_number + '.png', target_img * 255.)
    json_file = open('Target Data/' + file_number + '.json', 'w')
    json_file.write(json_string)
    json_file.close()


if __name__ == '__main__':
    params = parse_target()
    img = cv2.imread(params.input_file)
    num_targets = params.num_targets
    for i in range(0, num_targets):
        write_target_image_and_json(str(i), img)
    # from target_gen import create_target_image
    # targ = Target('d',
    #               Shape.Cross,
    #               alphanumeric_color=Color.White,
    #               shape_color=Color.Black,
    #               posx=0,
    #               posy=0,
    #               scale=15,
    #               rotation=0)
    # img = create_target_image(targ)
    # cv2.imshow('d', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()
