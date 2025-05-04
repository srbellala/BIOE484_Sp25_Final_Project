import numpy as np
import cv2

def Otsu(image, threshold, maxVal=255, technique = cv2.THRESH_OTSU): 
    """Performs Thresholding using the Otsu method from the cv2 library

    args: 
        image: Input image array (must be grayscale)
        threshold: Value of Threshold below and above which pixels will change according
        maxVal: Maximum Value that can be assigned to a pixel
        technique: default technique will be Otsu Thresholding

    returns: 
        thresImg: Thresholed image return by otsu algo
        threshVal: actual thresholded val used by otsu algo
    """

    if image is None: 
        raise ValueError("Input image is None")
    if image.shape==0: 
        raise ValueError("Input image has invalid shape")
    if len(image.shape) != 2: 
        raise ValueError("Input image has invalid shape")

    #convert image to CV_8UC1
    img_8U = cv2.convertScaleAbs(image)

    #Perform Thresholding with Otsu flag
    threshVal, threshImg = cv2.threshold(img_8U, threshold, maxVal, cv2.THRESH_BINARY+technique)



    return threshVal, threshImg

