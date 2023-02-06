import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class ModLines(Enum):
    BOTH = 1
    ONLY_LEFT = 2
    ONLY_RIGHT = 3
    NONE = 4

class PairOfLines:
    def __init__(self,left_line_x = None, right_line_x = None):
        self.left_line_x = left_line_x
        self.right_line_x = right_line_x

    def mod(self):
        if self.left_line_x is not None  and  self.right_line_x is not None :
            return ModLines.BOTH
        if self.left_line_x is not None  and  self.right_line_x is None :
            return ModLines.ONLY_LEFT
        if self.left_line_x is None  and  self.right_line_x is not None :
            return ModLines.ONLY_RIGHT
        if self.left_line_x is None  and  self.right_line_x is None :
            return ModLines.NONE


def calculate_histogram(frame):  
    """
        Prosledjuje mu se slika iz pticije perspektive, a on uzima donju polovinu pa je dosta robustan.
        Moze da detektuje sumove jer koristi sirinu od jedan piksel
    """
    histogram = np.sum(frame[int(frame.shape[0]/2):,:], axis=0)
 
    return histogram


def make_pair_of_lines(changed_perspective):
    """
        Vraca objekat klase PairOfLines koji u sebi sadrzi informacije sa histograma.
    """
    histogram = calculate_histogram(changed_perspective)

    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    if histogram[leftx_base] > 2000 and histogram[rightx_base] > 2000 :
        return PairOfLines(leftx_base,rightx_base)
    
    elif histogram[leftx_base] > 2000 and histogram[rightx_base] <= 2000 :
        return PairOfLines(leftx_base,None)
    
    elif histogram[leftx_base] <= 2000 and histogram[rightx_base] > 2000 :
        return PairOfLines(None,rightx_base)
    
    else:
        return PairOfLines(None, None)    

    