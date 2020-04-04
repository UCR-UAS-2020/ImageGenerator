import numpy as np
import cv2 as cv
import json

from proto import Target

currentNumber = 1


def write_to_json(target):
    data = {
        str(currentNumber): {
            "alphanumeric": target.alphanumeric,
            "shape": target.shape,
            "alphanumeric_color": target.color_alphanum,
            "shape_color": target.color_shape,
            "x": target.pos, # Replace with proper value
            "y": target.pos, # Replace with proper value
            "rotation": target.pos, # Replace with proper value
            "scale": target.scale
        }
    }

    print(json.dumps(data, indent = 2))
