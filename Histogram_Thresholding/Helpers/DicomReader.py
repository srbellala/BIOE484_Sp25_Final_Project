import pydicom
from pydicom import pixel_array
import matplotlib.pyplot as plt

def ReadDicom(Path):
    """Utility for reading in dicom data to generate dicom object/images. 

    args: 
        Path: image path to the dicom data

    returns: 
        dicomPixelArr: returns array of pixels for the dicom image
    """

    dicomObject = pydicom.dcmread(Path)
    dicomArr = pixel_array(dicomObject)

    #Check if pixel data exists
    if dicomArr is None: 
        raise AssertionError("dicom pixel array is None!")
    else:
        return dicomArr



    
