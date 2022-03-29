"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 315074963


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    if representation == 1:  # meaning the pic is gray scale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)
        return gray_img
    else:  # image is in RBG form but actually BGR
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return rgb


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    pic = imReadAndConvert(filename, representation)
    plt.imshow(pic)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    formula_YIQ = np.array([[0.299, 0.587, 0.114],
                            [0.59590059, -0.27455667, -0.32134392],
                            [0.21153661, -0.52273617, 0.31119955]])
    YIQ = imgRGB @ formula_YIQ.transpose()
    return YIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    formula_YIQ = np.array([[0.299, 0.587, 0.114],
                            [0.59590059, -0.27455667, -0.32134392],
                            [0.21153661, -0.52273617, 0.31119955]])
    YIQ = np.linalg.inv(formula_YIQ)
    RGB = imgYIQ @ YIQ.transpose()
    return RGB


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """


    # change range of imgOrig from [0, 1] to [0, 255] and normalize the image
    imgOrig_normalize = cv2.normalize(imgOrig, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    imgOrig_normalize = imgOrig_normalize.astype(np.uint8)

    # Calculate the old image histogram (range = [0, 255]), i used cv calchist
    # histOrg = cv2.calcHist([imgOrig_normalize], [0], None, [256], [0, 256])
    img_flat = imgOrig_normalize.ravel()
    hist = np.zeros(256)
    for val in img_flat:
        hist[val] += 1

    # Calculate the normalized Cumulative Sum (CumSum)
    C_sum = np.cumsum(hist)

    # Create a LookUpTable(LUT)
    look_ut = np.floor((C_sum / C_sum.max()) * 255)

    # Replace each intensity i with LUT[i]
    imgEq = np.zeros_like(imgOrig, dtype=float)
    for i in range(256):
        imgEq[imgOrig_normalize == i] = int(look_ut[i])

    # Calculate the new image histogram (range = [0, 255])
    histEQ = np.zeros(256)
    for val in range(256):
        histEQ[val] = np.count_nonzero(imgEq == val)

    # norm imgEQ from range [0, 255] to range [0, 1]
    imgEq = imgEq / 255.0

    return imgEq, hist, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
