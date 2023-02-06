import cv2
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum

def sliding_window(changed_perspective, starting_x_cord):
    """
        Pronalazi liniju na slici pocevsi od zadate x cord. Iscrtava pravougaonike na zadatoj slici.
    """
    width = changed_perspective.shape[1]
    height = changed_perspective.shape[0]

    no_of_windows = 10
    margin = int((1/12) * width)  # Window width is +/- margin
    minpix = int((1/24) * width)

    window_height = int(height/no_of_windows)


    nonzero = changed_perspective.nonzero() 
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])    
    # prvi beli piksel na slici ima kordinate (x,y) = (nonzerox[0],nonzeroy[0])

    lane_inds = []
    curr_x = starting_x_cord

    for window in range(no_of_windows):
        window_y_low = height - (window + 1) * window_height
        window_y_high = height - window * window_height
        window_x_left = curr_x - margin
        window_x_right = curr_x + margin
        cv2.rectangle(changed_perspective,
            (window_x_left, window_y_low),(window_x_right,window_y_high), 255, 2)

        
        #indeksi niza nonzero koji pripadaju trenutnom windowu, 
        # nonzerox[good_inds[0]] - x kordinata prvog piksela unutar prozora
        good_inds = ((nonzeroy >= window_y_low) & (nonzeroy < window_y_high) & 
                          (nonzerox >= window_x_left) & (nonzerox < window_x_right)).nonzero()[0]


        lane_inds.append(good_inds)

        # Ako detektujemo isprekidanu liniju ne zelimo da sliding window skrene zbog malog suma
        if len(good_inds) > minpix:
            curr_x = int(np.mean(nonzerox[good_inds]))
        
    
    lane_inds = np.concatenate(lane_inds)

    # iscrtava pronadjenu liniju
    #for i in lane_inds:
    #    black_img[nonzeroy[i]][nonzerox[i]] = 255

    x_cord = nonzerox[lane_inds]
    y_cord = nonzeroy[lane_inds]

    fited_line = np.polyfit(y_cord,x_cord,2)

    ploty = np.linspace(0,height-1,height)
    ftited_x_cord = fited_line[0]*ploty**2 + fited_line[1]*ploty + fited_line[2]

    
    for i in range(0,len(ploty) - 1):
        if int(ftited_x_cord[i]) > 0 and int(ftited_x_cord[i]) < width - 1 :
            changed_perspective[int(ploty[i])][int(ftited_x_cord[i])] = 255
    
    return fited_line