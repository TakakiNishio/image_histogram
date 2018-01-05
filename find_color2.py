import cv2
import numpy as np
import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
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
    dominant_color = None

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
            dominant_color = (b, g, r)
    return dominant_color


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

cv_img = cv2.imread(args["image"])

crop_param = 4
height, width = cv_img.shape[0], cv_img.shape[1]
center = height/2, width/2

print center[0]
print center[1]

top = center[0] - height/crop_param
bottom = center[0]

left = center[1] - width/crop_param
right = center[1] + width/crop_param

cropped_img = cv_img[top:bottom, left:right]

pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))

# img = Image.open(args["image"])

dominant_color = get_dominant_color(pil_img)

print dominant_color

size = 100, 100, 3
color_display = np.zeros(size, dtype=np.uint8)
color_display[:] = dominant_color

cv2.imshow("dominant color", color_display)
cv2.imshow("original image", cv_img)
cv2.imshow("cropped image", cropped_img)
cv2.waitKey(0)

# crop_param = 4
# height, width = img.shape[0], img.shape[1]
# center = height/2, width/2

# print center[0]
# print center[1]

# top = center[0] - height/crop_param
# bottom = center[0]

# left = center[1] - width/crop_param
# right = center[1] + width/crop_param

# img = img[top:bottom, left:right]

# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
# clt = KMeans(n_clusters=3) #cluster number
# clt.fit(img)

# hist = find_histogram(clt)
# bar = plot_colors2(hist, clt.cluster_centers_)

# plt.axis("off")
# plt.imshow(bar)
# plt.show()
