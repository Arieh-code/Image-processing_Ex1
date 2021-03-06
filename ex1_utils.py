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

    if len(imOrig.shape) == 3:
        imYIQ = transformRGB2YIQ(imOrig)
        imY = imYIQ[:, :, 0].copy()  # take only the y chanel
    else:
        imY = imOrig
    histOrig = np.histogram(imY.flatten(), bins=256)[0]
    # finding the best center
    Z_borders = []
    Q_level = []
    # head start, all the intervals are in the same length
    z = np.arange(0, 256, round(256 / nQuant))
    z = np.append(z, [255])
    Z_borders.append(z.copy())
    # find Q using weighted average on the histogram
    q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=histOrig[z[k]: z[k + 1] + 1]) for k in range(len(z) - 1)]
    q = np.round(q).astype(int)
    Q_level.append(q.copy())

    # finding the best nQuant center in nIter steps or when error is minimum
    for n in range(nIter):
        # finding Z using the formula from lecture
        z = np.array([round((q[i - 1] + q[i]) / 2) for i in range(1, len(q))]).astype(int)
        z = np.concatenate(([0], z, [255]))
        if (Z_borders[-1] == z).all():  # break if nothing changed
            break
        Z_borders.append(z.copy())

        # fixing q in loop and adding it to Q array
        q = [np.average(np.arange(z[k], z[k + 1] + 1), weights=histOrig[z[k]: z[k + 1] + 1]) for k in
             range(len(z) - 1)]
        q = np.round(q).astype(int)
        Q_level.append(q.copy())

    image_history = [imOrig.copy()]
    MSE = []
    for i in range(len(Z_borders)):
        arrayQuantize = np.array([Q_level[i][k] for k in range(len(Q_level[i])) for x in range(Z_borders[i][k], Z_borders[i][k + 1])])
        if len(imOrig.shape) == 3:
            q_img, e = update_image(imY, histOrig, imYIQ, arrayQuantize)
        else:
            q_img, e = update_image(imY, histOrig, [], arrayQuantize)
        image_history.append(q_img)
        MSE.append(e)

    return image_history, MSE


def update_image(imOrig: np.ndarray, histOrig: np.ndarray, yiqIm: np.ndarray, arrayQuantize: np.ndarray) -> (
        np.ndarray, float):
    """
        update the quantization on the original image
        :return: returning the resulting image and the MSE.
    """
    imageQ = np.interp(imOrig, np.linspace(0, 1, 255), arrayQuantize)
    curr_hist = np.histogram(imageQ, bins=256)[0]
    err = np.sqrt(np.sum((histOrig.astype('float') - curr_hist.astype('float')) ** 2)) / float(
        imOrig.shape[0] * imOrig.shape[1])
    if len(yiqIm):  # if the original image is RGB
        yiqIm[:, :, 0] = imageQ / 255
        return transformYIQ2RGB(yiqIm), err
    return imageQ, err
