import cv2

def GuassFilter(image, size, sigma=1): 
    """Performs a guassian filter on the image

    args: 
        image: grayscale image to perform the guassian filter on

    returns:
        SmoothImg: smoothed image 
    """

    size_tuple = (size,size)

    if image is None: 
        raise ValueError("input image is None")
    if image.shape==0: 
        raise ValueError("Input image has invalid shape")
    if len(image.shape) != 2: 
        raise ValueError("Input image has invalid shape")
    if size%2==0: 
        raise ValueError("Kernel size must be odd number")
    
    blurImg = cv2.GaussianBlur(image, size_tuple, sigma)

    return blurImg
