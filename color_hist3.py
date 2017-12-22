import numpy as np
import cv2
from time import clock
import sys
import argparse
from matplotlib import pyplot as plt


def RGB_hist(img):

    b, g, r = img[:,:,0], img[:,:,1], img[:,:,2]

    hist_r = cv2.calcHist([r],[0],None,[256],[0,256])
    hist_g = cv2.calcHist([g],[0],None,[256],[0,256])
    hist_b = cv2.calcHist([b],[0],None,[256],[0,256])

    plt.xlim(0, 255)
    plt.plot(hist_r, "-r", label="Red")
    plt.plot(hist_g, "-g", label="Green")
    plt.plot(hist_b, "-b", label="Blue")
    plt.xlabel("Pixel value", fontsize=20)
    plt.ylabel("Number of pixels", fontsize=20)
    plt.legend()
    plt.grid()
    plt.pause(.05)


if __name__ == '__main__':

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required = True, help = "Path to the image")
    args = vars(ap.parse_args())

    crop_param = 4

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("hist", cv2.WINDOW_NORMAL)

    image = cv2.imread(args["image"])
    height, width = image.shape[0], image.shape[1]
    center = height/2, width/2

    print center[0]
    print center[1]

    top = center[0] - height/crop_param
    #bottom = center[0] + height/2
    bottom = center[0]

    left = center[1] - width/crop_param
    right = center[1] + width/crop_param

    image = image[top:bottom, left:right]
    cv2.imshow("image", image)

    hsv_map = np.zeros((180, 256, 3), np.uint8)
    h, s = np.indices(hsv_map.shape[:2])
    hsv_map[:,:,0] = h
    hsv_map[:,:,1] = s
    hsv_map[:,:,2] = 255
    hsv_map = cv2.cvtColor(hsv_map, cv2.COLOR_HSV2BGR)

    RGB_hist(image)

    hist_scale = 50
    max_hist_scale = 300

    def set_scale(val):
        global hist_scale
        hist_scale = val

    cv2.createTrackbar('scale', 'hist', hist_scale, max_hist_scale, set_scale)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # dark = hsv[...,2] < 32
    # hsv[dark] = 0
    ho = cv2.calcHist( [hsv], [0, 1], None, [180, 256], [0, 180, 0, 256] )

    while(True):

        h = np.clip(ho*0.005*hist_scale, 0, 1)
        vis = hsv_map*h[:,:,np.newaxis] / 255.0
        print np.max(vis)
        cv2.imshow('hist', vis)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
