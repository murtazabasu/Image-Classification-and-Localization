import cv2
import torch
import numpy as np


class LocalizeParasite():
    """
    Module for localizing parasites by drawing contour on the image.
    The module applies a series of computer vision techniques on the input image.
    --initialization parameter--
    :param colorCode: cv2 color code to convert input image to target format.
    :param thresholdType: cv2 threshold type for thresholding input image.
    :param thresholdVal: min threshold value to classify pixel values.
    :param thresoldMaxVal: maximum value which is assigned to pixel values exceeding the threshold.
    :param cannyLowerThreshold: threshold that rejects pixel gradient value lower than this value.
    :param cannyUpperThreshold: threshold that accepts pixel as an edge wjose radient value is above this value. 
    :param contourMode: (contour retrieval mode) returns only extreme outer flag, leaves all child contour behind.
    :param contourApproxMethod: (contour approx method) stores the (x,y) coordinates of the boundary of a shape based on selected method.  
    :param contourIdx: index to select a particular contour -1 denotes last index.
    :param contourRGB: RGB color code for contour. 
    :param lineThickness: contour thickness.      
    """
    def __init__(
        self, 
        colorCode:int=cv2.COLOR_RGB2BGR, 
        thresholdType:int=cv2.THRESH_BINARY, 
        thresholdVal:int=110, 
        thresholdMaxVal:int=175, 
        cannyLowerThreshold:int=10, 
        cannyUpperThreshold:int=100, 
        contourMode:int=cv2.RETR_EXTERNAL,  
        contourApproxMethod:int=cv2.CHAIN_APPROX_SIMPLE, 
        contourIdx:int=-1, 
        contourRGB:tuple=(0,0,0), 
        lineThickness:int=1
    ):
        
        self.colorCode = colorCode
        self.thresholdType = thresholdType
        self.thresholdVal = thresholdVal
        self.thresholdMaxVal = thresholdMaxVal
        self.cannyLowerThreshold = cannyLowerThreshold
        self.cannyUpperThreshold = cannyUpperThreshold
        self.contourMode = contourMode
        self.contourApproxMethod = contourApproxMethod
        self.contourIdx = contourIdx
        self.contourRGB = contourRGB
        self.lineThickness = lineThickness
    
    def prepareImage(self, inputImage:torch.Tensor):
        """
        Processes the input pytorch tensor into numpy uint8 type
        to be proessed by the cv2 module.
        :param inputImage: input pytorch image tensor.
        :returns numpy uint8 array:
        """
        image = np.moveaxis(inputImage.cpu().numpy(), 0, -1)
        
        # normalize pixels between 0 and 255
        image *= 255.0 / image.max()
        image = image.astype(np.uint8)
        return image

    def filter_contours(self, contours, hierachy, softing=0.2):      
        """
        Method to filter contour to remove contours that are not useful with
        For e.g. contours at the image boundary.
        :param contours: list of coordinates.
        :param hierachy: contains information about the image topology, it is an array of 4 element 
        [Next, Previous, First_Child, Parent].
        :returns: filtered contour.
        """
        areas = []
        filtered_contours = []
        l = []          
        for e, h in enumerate(hierachy[0]):
            # In the hierarchy array scan the Parent column (4th column) 
            # to see whether it has no parent contours (-1) and append them.
            if h[3] == -1:
                l.append(e)

        contours_ = []
        for o in l:
            # compute contour area. 
            areas.append(cv2.contourArea(contours[o]))
            # append contours that as no parent cotour
            contours_.append(contours[o])

        # compute a threshold based on contour area.
        contThreshold = np.mean(areas) + softing*np.max(areas)
        
        for cont in contours_:
            if cv2.contourArea(cont) > 10:
                # append those contour that has contour area > threshold
                filtered_contours.append(cont) 

        return filtered_contours


    def getContourImage(self, inputImage:np.uint8, applyThresholding:bool=True, filterContours:bool=False):
        """
        Applies a series of transformation techniques from cv2 tool
        generate and draws contour on the input image.
        :param inputImage: image numpy array of type uint8.
        :returns: returns image with contour.
        """
        # convert an image from one color space to another. 
        image = cv2.cvtColor(inputImage, self.colorCode)
       
        # Thresholding is a technique in OpenCV, which is the assignment of pixel values 
        # in relation to the threshold value provided. In thresholding, each pixel value 
        # is compared with the threshold value. If the pixel value is smaller than the threshold, 
        # it is set to 0, otherwise, it is set to a maximum value (generally 255). 
        # Thresholding is a very popular segmentation technique, used for separating an 
        # object considered as a foreground from its background.

        thresholdImage = image.copy()

        if applyThresholding:
            _, thresholdImage = cv2.threshold(
                image, 
                self.thresholdVal, 
                self.thresholdMaxVal,
                self.thresholdType
            )

        # Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images
        # that uses a multi-stage algorithm to detect a wide range of edges in images. The steps are as followed. 
        # 1. Reduce Noise using Gaussian Smoothing.
        # 2. Compute image gradient using Sobel filter.
        # 3. Apply Non-Max Suppression or NMS to just jeep the local maxima
        # 4. Finally, apply Hysteresis thresholding which that 2 threshold values T_upper and T_lower which is used in the Canny() function.
        cannyImage = cv2.Canny(
            thresholdImage, 
            self.cannyLowerThreshold, 
            self.cannyUpperThreshold
        )
        
        # findContour function that helps in extracting the contours from the image. 
        # It works best on binary images, so we should first apply thresholding techniques.
        contours_, hierachy_ = cv2.findContours(
            cannyImage,
            self.contourMode,
            self.contourApproxMethod
        )

        if filterContours:
            contours_ = self.filter_contours(
                contours=contours_, 
                hierachy=hierachy_
            )

        # To draw the contours, cv.drawContours function is used. 
        # It can also be used to draw any shape provided you have its boundary points. 
        # Its first argument is source image, second argument is the contours which should be passed as a Python list, 
        # third argument is index of contours (useful when drawing individual contour. To draw all contours, pass -1) 
        # and remaining arguments are color, thickness etc.
        drawContours = cv2.drawContours(
            image, 
            contours_, 
            self.contourIdx, 
            self.contourRGB, 
            self.lineThickness
        )
        return image
