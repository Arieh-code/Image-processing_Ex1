# Image-Processing Ex1


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Content</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#files-to-submit">Files To Submit</a></li>
    <li><a href="#function-details">Function Details</a></li>
    <li><a href="#test-images">Test Images</a></li>
    <li><a href="#question-4">Question 4</a></li>
    <li><a href="#languages-and-tools">Languages and Tools</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>

----------------

<!-- ABOUT THE PROJECT -->
# About The Project
*_Image-Processing Ex1:_*

In this assignment our main task was to implement Histogram Equalization and Optimal image quantization. 
through out this assignment I go to learn some of pythongs basics syntax and image processing facitilites using open cv.

* Loading grayscale and RGB image representations.
* Displaying figures and images.
* Transforming RGB color images back and forth from the YIQ color space.
* Performing intensity transformations: histogram equalization.
* Performing optimal quantization


``` Version Python 3.10.4```

``` Pycharm```

## Files To Submit

* ex1_main - This file runs all the code
* ex1_utils - This file has all the functions 
* gamma - This file is in charge of gamma correction 
* Ex1 - This is the assignment pdf 
* images 
* Readme

---------------------

## Function Details

The discription of each function is the same discription we are given in the assingment from the [pdf](https://github.com/Arieh-code/Image-processing_Ex1/blob/master/Ex1.pdf)
 
Displaying an image - function that utilizes imReadAndConvert to display a given image file in a given representation.

The function should have the following interface:

```python
def imDisplay(filename:str, representation:int)->None:
    """
    Reads an image as RGB or GRAY_SCALE and displays it`
    :param filename: The path to the image
    :param representation: grayscale(1) or RGB(2)
    :return: None
    """
```

Transforming an RGB image to YIQ color space - Write two functions that transform an RGB image into the YIQ color space (mentioned in the lecture)
and vice versa. Given the red (R), green (G), and blue (B) pixel components of an RGB color image,
the corresponding luminance (Y), and the chromaticity components (I and Q) in the YIQ color space are
linearly related as follows:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/JcfrGkh/Untitled.png" alt="Untitled" border="0"></a>

The two functions should have the following interfaces:
```python
def transformRGB2YIQ(imRGB:np.ndarray)->np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
```

```python
def transformYIQ2RGB(imYIQ:np.ndarray)->np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
```

Histogram equalization - Write a function that performs histogram equalization of a given grayscale or RGB image. The function
should also display the input and the equalized output image. 

The function should have the following interface:

```python
def histogramEqualize(imOrig:np.ndarray)->(np.ndarray,np.ndarray,np.ndarray):
    """
    Equalizes the histogram of an image
    :param imgOrig: Original image
    :return: (imgEq,histOrg,histEQ)
    """
```

Optimal image quantization - Write a function that performs optimal quantization of a given grayscale or RGB image. The function
should return:

* A list of the quantized image in each iteration
* A list of the MSE error in each iteration

The function should have the following interface:

```python
def quantizeImage(imOrig:np.ndarray, nQuant:int, nIter:int)->(List[np.ndarray],List[float]):
    """
    Quantized an image in to **nQuant** colors
    :param imOrig: The original image (RGB or Gray scale)
    :param nQuant: Number of colors to quantize the image to
    :param nIter: Number of optimization loops
    :return: (List[qImage_i],List[error_i])
    """
```

Gamma Correction - function that performs gamma correction on an image with a given γ.
For this task, you’ll be using the OpenCV functions createTrackbar to create the slider and display
it, since it’s OpenCV’s functions, the image will have to be represented as BGR.

```python
def gammaDisplay(img_path:str, rep:int)->None:
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
```

---------------------

## Test Images

* TestImg1 

![testImg1](https://user-images.githubusercontent.com/68643157/161394523-9625b97b-e0b3-4e80-a627-a5884dbf21b2.jpg)

* TestImg2

![testImg2](https://user-images.githubusercontent.com/68643157/161394677-4abb6ecd-73b5-45a5-8cee-991d10d736fd.jpg)

*The reason I chose these images is because they represent a lot of images from our day to day. I wanted to test my code on, 
 on weather Histogram equalization would work well on a dark background picture and weather quantization would work on a really colorful image.*

---------------------

## Question 4

### Answer to question from section 4.5:

*If a division will have a grey level segment with no pixels, procedure will crash because we will not be able to calculate the weighted average for this segment 
because we need to divide by the number of pixels for this segment, but in this case the number is zero.*

---------------------

## Languages and Tools



  <div align="center">
  
 <code><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png"></code> 
 <code><img height="40" width="80" src="https://matplotlib.org/_static/logo2_compressed.svg"/></code>
 <code><img height="40" width="40" src="https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/PyCharm_Icon.svg/1024px-PyCharm_Icon.svg.png"/></code>
 <code><img height="40" height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/git/git.png"></code>
 <code><img height="40" height="40" src="https://raw.githubusercontent.com/github/explore/80688e429a7d4ef2fca1e82350fe8e3517d3494d/topics/terminal/terminal.png"></code>
  </div>


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Python](https://www.python.org/)
* [Matplotlib](https://matplotlib.org/)
* [Git](https://git-scm.com/)
* [Pycharm](https://www.jetbrains.com/pycharm/)
* [Git-scm](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)


