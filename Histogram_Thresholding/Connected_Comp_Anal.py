import numpy as np
import argparse
import cv2

def Conn_Comp_Anal(image): 
    """Perform connected component analysis to label regions of interest

    args: 
        image: binary threshold image

    returns: 
        numlabels: total number of unique labels
        labels: A mask named labels. Has same spaitial dimensions as our input thresh image. 
                Each location in labels has integer ID that corresponds to connnected component where the pixel belongs
        stats: stats on each connected component, including the bounding box coordinates and area in pixels
        centroids: center coordinates; (x,y)-coordinates of each connected component
    """

    #construct argument parser
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to input image")
    ap.add_argument("-c", "--connectivity", type=int, default = 4, help = "connectivity for connected component analysis")
    args=vars(ap.parse_args())

    #Perform the Connected components Analysis
    thresh_labeled = cv2.connectedComponentsWithStats(image, args["Connectivity"], cv2.CV_32S)

    #Retrieve the 4-tuple from the connected component analysis
    (numlabels, labels, stats, centroids) = thresh_labeled

    return numlabels, labels, stats, centroids