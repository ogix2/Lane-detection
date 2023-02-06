import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

class PreprocesMod(Enum):
    BLACK_LINES = 1
    WHITE_LINES = 2

def preprocesing(frame):
    
    mod = PreprocesMod.WHITE_LINES

    if (mod == PreprocesMod.BLACK_LINES):
        
        new_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        _ , new_img = cv2.threshold(new_img, 150, 255, cv2.THRESH_BINARY_INV)

        return new_img

    elif (mod == PreprocesMod.WHITE_LINES):

        new_img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        _ , new_img = cv2.threshold(new_img, 185, 255, cv2.THRESH_BINARY)

        return new_img

