import cv2
import numpy as np
import LineDetection as LD
import SlidingWindow
import Preprocesing


obj = LD.LineDetection([[0.0,0.0],[1.0,0.0],[0.7,0.5],[0.3,0.5]])


cap = cv2.VideoCapture("Resources/video.mp4") 

inic = False
while (cap.isOpened()):
    _, frame = cap.read()

    if inic == False:
        obj.update_frame(frame)
        obj.update_inic_lines()
        obj.inic_mod = LD.ModInic.WORK
        inic = True
        cv2.destroyAllWindows()
    else:
        obj.update_frame(frame)
        print(obj.center_offset)
    
    if cv2.waitKey(15) & 0xFF == ord('q'):  # waikey() ti pravi pauzu izmedju framova
        break

