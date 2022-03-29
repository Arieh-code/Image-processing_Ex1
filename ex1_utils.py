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

    RGB = False
    if len(imgOrig.shape) == 3:  # if the image is RGB
        RGB = True
        yiqIm = transformRGB2YIQ(imgOrig)
        imgOrig = yiqIm[:, :, 0]
    # change image from [0,1] t0 [0,255]
    imgOrig = cv2.normalize(imgOrig, None, 0, 255, cv2.NORM_MINMAX)
    imgOrig = imgOrig.astype('uint8')

    # flatten the image
    histOrig = np.histogram(imgOrig.flatten(), bins=256)[0]
    C_sum = np.cumsum(histOrig)
    imgNew = C_sum[imgOrig]
    imgNew = cv2.normalize(imgNew, None, 0, 255, cv2.NORM_MINMAX)
    imgNew = imgNew.astype('uint8')
    histNew = np.histogram(imgNew.flatten(), bins=256)[0]

    # if the image was in color then I need to transform back to from yiq to rgb
    if RGB:
        yiqIm[:, :, 0] = imgNew / (imgNew.max() - imgNew.min())
        imgNew = transformYIQ2RGB(yiqIm)

    return imgNew, histOrig, histNew


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass



