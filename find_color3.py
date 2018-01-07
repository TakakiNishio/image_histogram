import cv2
import numpy as np
import argparse
import colorsys

from PIL import Image

def get_dominant_color(image):
    """
    Find a PIL image's dominant color, returning an (r, g, b) tuple.
    """
    image = image.convert('RGBA')
    # Shrink the image, so we don't spend too long analysing color
    # frequencies. We're not interpolating so should be quick.
    image.thumbnail((200, 200))
    max_score = None
    dominant_color1 = None
    dominant_color2 = None

    for count, (r, g, b, a) in image.getcolors(image.size[0] * image.size[1]):
        # Skip 100% transparent pixels
        if a == 0:
            continue
        # Get color saturation, 0-1
        saturation = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)[1]
        # Calculate luminance - integer YUV conversion from
        # http://en.wikipedia.org/wiki/YUV
        y = min(abs(r * 2104 + g * 4130 + b * 802 + 4096 + 131072) >> 13, 235)
        # Rescale luminance from 16-235 to 0-1
        y = (y - 16.0) / (235 - 16)
        # Ignore the brightest colors
        if y > 0.9:
            continue
        # Calculate the score, preferring highly saturated colors.
        # Add 0.1 to the saturation so we don't completely ignore grayscale
        # colors by multiplying the count by zero, but still give them a low
        # weight.
        score = (saturation + 0.1) * count
        if score > max_score:
            max_score = score
            dominant_color2 = dominant_color1
            dominant_color1 = (b, g, r)

    return dominant_color1, dominant_color2


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    # load an image
    cv_img = cv2.imread(args["image"])

    # crop image
    crop_param = 4
    height, width = cv_img.shape[0], cv_img.shape[1]
    center = height/2, width/2

    top = center[0] - height/crop_param
    bottom = center[0]

    left = center[1] - width/crop_param
    right = center[1] + width/crop_param

    cropped_img = cv_img[top:bottom, left:right]

    # convert the image into PIL image format
    pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

    # detect the dominant color
    dominant_color1, dominant_color2 = get_dominant_color(pil_img)

    # calculate average color
    average_color = [cropped_img[:, :, i].mean() for i in range(cropped_img.shape[-1])]

    print "dominant color"
    print dominant_color1
    print dominant_color2

    print
    print "average color"
    print average_color


    # display doninant color
    size = 100, 100, 3
    dominant_color_display1 = np.zeros(size, dtype=np.uint8)
    dominant_color_display1[:] = dominant_color1

    dominant_color_display2 = np.zeros(size, dtype=np.uint8)
    dominant_color_display2[:] = dominant_color2

    # display average color
    average_color_display = np.zeros(size, dtype=np.uint8)
    average_color_display[:] = average_color

    cv2.imshow("original image", cv_img)
    cv2.imshow("cropped image", cropped_img)
    cv2.imshow("color1", dominant_color_display1)
    cv2.imshow("color2", dominant_color_display2)
    cv2.imshow("average", average_color_display)
    cv2.waitKey(0)
