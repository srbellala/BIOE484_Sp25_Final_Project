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


    #Perform the Connected components Analysis
    thresh_labeled = cv2.connectedComponents(image)

    #Retrieve the tuple from the connected component analysis
    (numlabels, labels) = thresh_labeled

    return numlabels, labels