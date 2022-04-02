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
from __future__ import print_function
import numpy as np
import argparse
import cv2 as cv

from ex1_utils import LOAD_GRAY_SCALE

title_window = 'Gamma correction'


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    global img
    if rep == LOAD_GRAY_SCALE:  # for gray image
        img = cv.imread(img_path, 1)
    else:  # rep = LOAD_RGB
        img = cv.imread(img_path, 2)
    trackbar_name = 'Gamma'
    cv.namedWindow(title_window)
    cv.createTrackbar(trackbar_name, title_window, 0, 200, on_trackbar)
    while True:
        gamma = cv.getTrackbarPos(trackbar_name, title_window)
        gamma = gamma / 100 * (2 - 0.01)
        gamma = 0.01 if gamma == 0 else gamma
        newImg = adjust_gamma(img, gamma)
        cv.imshow(title_window, newImg)
        k = cv.waitKey(1000)
        if k == 27:  # esc button
            break
        if cv.getWindowProperty(title_window, cv.WND_PROP_VISIBLE) < 1:
            break
    cv.destroyAllWindows()


def on_trackbar(val):
    gamma = float(val) / 100
    invGamma = 1000 if gamma == 0 else 1.0 / gamma
    max_ = 255
    gammaTable = np.array([((i / float(max_)) ** invGamma) * max_
                           for i in np.arange(0, max_ + 1)]).astype("uint8")
    img_ = cv.LUT(img, gammaTable)
    cv.imshow(title_window, img_)


# I read about this code on this website "https://pyimagesearch.com/2015/10/05/opencv-gamma-correction/"
def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """
        Gamma correction
        :param image: the original image
        :param gamma: the gamma number
        :return: the new image after the gamma operation
        """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    return cv.LUT(image, table)



