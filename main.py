import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # src = cv.imread('beach.jpg')
    # src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    # dst = cv.equalizeHist(src)
    # cv.imshow('Source image', src)
    # cv.imshow('Equalized Image', dst)
    # cv.waitKey()


    img = cv.imread('beach.jpg')

    img_yuv = cv.cvtColor(img, cv.COLOR_BGR2YUV)

    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_output = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)

    cv.imshow('Color input image', img)
    cv.imshow('Histogram equalized', img_output)

    cv.waitKey(0)