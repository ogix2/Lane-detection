import cv2
import numpy as np
from enum import Enum
import Preprocesing
import Histogram
import SlidingWindow

class ModInic(Enum):
    INIC = 1
    WORK = 2

class LineDetection:
    def __init__(self,ROI_param):
        """
            ROI_param - lista normalizovanih tacaka trapeza [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]  
                            * smer kazaljke na satu
                            * (0,0) je donje levo teme slike
        """    
        self.inic_mod = ModInic.INIC
        self.ROI_param = ROI_param

        self.curr_frame = None
        
        #Ostace None sve dok kamere ne detektuje linije. 
        self.radius_of_curvatue = None
        self.center_offset = None

        #X kordinate leve i desne linije kada je auto u centru.
        self.left_line_inic_x = None
        self.right_line_inic_x = None

    def update_frame(self, frame):
        self.curr_frame = frame

        if self.inic_mod == ModInic.INIC:
            cv2.imshow("ROI",self.show_ROI())
            cv2.imshow("Preprocesing", Preprocesing.preprocesing(frame))
        else:
            self.pipeline()
    
    
    def update_inic_lines(self):
        """
            Metoda koja se poziva kada postavimo auto u centar trake.
        """
        if self.curr_frame is None:
            raise Exception("Updater_inic se poziva posle update frame.")

        prepocesed_frame = Preprocesing.preprocesing(self.curr_frame.copy())

        changed_perspective = self.perspective_transform(prepocesed_frame)

        pair_of_lines = Histogram.make_pair_of_lines(changed_perspective)

        draw_sliding_window = changed_perspective.copy() 
        left_fited_line, right_fited_line = self.call_sliding_window(draw_sliding_window, pair_of_lines)

        height = self.curr_frame.shape[0]
        self.left_line_inic_x = left_fited_line[0]*height**2 + left_fited_line[1]*height + left_fited_line[2]
        self.right_line_inic_x = right_fited_line[0]*height**2 + right_fited_line[1]*height + right_fited_line[2]

        plot = True
        if plot == True:
            print("X kordinata inicijalne leve linije: ", self.left_line_inic_x)
            print("Y kordinata inicijalne leve linije: ", self.right_line_inic_x)
            cv2.imshow("original", self.curr_frame)
            cv2.imshow("changed perspective",changed_perspective)
            cv2.imshow("sliding window", draw_sliding_window)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


    def calculate_ROI_dots(self):
        if self.curr_frame is None:
            raise Exception("Enter the image first.")
        
        width = self.curr_frame.shape[1]
        height = self.curr_frame.shape[0]
        [x1,y1],[x2,y2],[x3,y3],[x4,y4] = self.ROI_param    

        x1 = int(x1 * width)
        x2 = int(x2 * width)
        x3 = int(x3 * width)
        x4 = int(x4 * width)
        y1 = int((1.0 - y1) * height)
        y2 = int((1.0 - y2) * height)
        y3 = int((1.0 - y3) * height)
        y4 = int((1.0 - y4) * height)

        
        return [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        
    
    def show_ROI(self):
        
        polygons = np.int32([self.calculate_ROI_dots()])

        this_image = cv2.polylines(self.curr_frame.copy(), polygons, True, (147,20,255), 3)

        return this_image


    def perspective_transform(self, preprocesed_frame):

        ROI_points = np.float32(self.calculate_ROI_dots()) 
        
        width = self.curr_frame.shape[1]
        height = self.curr_frame.shape[0]
        
        padding = int(0.3 * width)
        desired_ROI_points = np.float32([
            [padding, height], # Bottom-left corner   
            [width-padding, height], # Bottom-right corner  
            [width-padding, 0], # Top-right corner
            [padding, 0] # Top-left corner
            ]) 
        
        
        transformation_matrix = cv2.getPerspectiveTransform(
            ROI_points, desired_ROI_points)

       
        inv_transformation_matrix = cv2.getPerspectiveTransform(
            desired_ROI_points, ROI_points)

        # Perform the transform using the transformation matrix
        warped_frame = cv2.warpPerspective(
            preprocesed_frame, transformation_matrix, (width,height), flags=(
            cv2.INTER_LINEAR)) 
        
        return warped_frame

    def call_sliding_window(self, changed_perspective, pair_of_lines):
        """
            Poziva funkciju sliding_window onoliko puta koliko je linija detektovano.
            Iscrtava pravougaonike direktno na changed_perspective
        """
        
        if pair_of_lines.mod() == Histogram.ModLines.NONE :
            return (None,None)
        
        elif pair_of_lines.mod() == Histogram.ModLines.BOTH :
            
            left_fited_line = SlidingWindow.sliding_window(changed_perspective, pair_of_lines.left_line_x)
            right_fited_line = SlidingWindow.sliding_window(changed_perspective, pair_of_lines.right_line_x)

            return (left_fited_line,right_fited_line)
        
        elif pair_of_lines.mod() == Histogram.ModLines.ONLY_LEFT :
            
            left_fited_line = SlidingWindow.sliding_window(changed_perspective, pair_of_lines.left_line_x)

            return (left_fited_line, None)
        
        elif pair_of_lines.mod() == Histogram.ModLines.ONLY_RIGHT :
            
            right_fited_line = SlidingWindow.sliding_window(changed_perspective, pair_of_lines.right_line_x)

            return (None, right_fited_line)

        else :
            raise Exception("Lose poslat argument pair_of_lines.")
        
    def update_car_offset(self,left_fited_line, right_fited_line, pair_of_lines) :
        """
            Racuna odstojanje od centra trake. Pozitivno ako je auto levo.
            const - Parametar koji se empirijski odredjuje, na kojoj visini od dna da se racuna.

        """
        height = self.curr_frame.shape[0]
        width = self.curr_frame.shape[1]
        car_position = width / 2
        const = 0.08
        
        if pair_of_lines.mod() == Histogram.ModLines.NONE :
            return  

        elif pair_of_lines.mod() == Histogram.ModLines.BOTH :
            
            y = (1.0 - const) * height
            x_left = left_fited_line[0]*y**2 + left_fited_line[1]*y + left_fited_line[2] 
            x_right = right_fited_line[0]*y**2 + right_fited_line[1]*y + right_fited_line[2]

            center_lane = (x_left + x_right) / 2
            
            self.center_offset = center_lane - car_position
            
        elif pair_of_lines.mod() == Histogram.ModLines.ONLY_LEFT :
            
            y = (1.0 - const) * height
            x_left = left_fited_line[0]*y**2 + left_fited_line[1]*y + left_fited_line[2] 
            
            self.center_offset =  x_left - self.left_line_inic_x 
        
        elif pair_of_lines.mod() == Histogram.ModLines.ONLY_RIGHT :
            
            y = (1.0 - const) * height
            x_right = right_fited_line[0]*y**2 + right_fited_line[1]*y + right_fited_line[2]

            self.center_offset = x_right - self.right_line_inic_x

        else :
            raise Exception("Lose poslat argument pair_of_lines.") 
     
    def update_radius_of_curvature(self,left_fited_line, right_fited_line, pair_of_lines):
        
        if pair_of_lines.mod() == Histogram.ModLines.NONE :
           
           print("KURCINA")

        elif pair_of_lines.mod() == Histogram.ModLines.BOTH :
            
            print("KURCINA")
        
        elif pair_of_lines.mod() == Histogram.ModLines.ONLY_LEFT :
            
            print("KURCINA")
        
        elif pair_of_lines.mod() == Histogram.ModLines.ONLY_RIGHT :
            
            print("KURCINA")
        else :
            raise Exception("Lose poslat argument pair_of_lines.") 
        

    def pipeline(self):
        """
            Prolazi kroz proces detekcije linija.
            Komunikacija izmedju funkcija se vrsi tako sto Histogram odredi u kom je modu trenuti frame,
            pa se zatim objekat klase PairOfLines prosledjuje svim funkcijama kako bi znale u kom modu rade.
        """
        if self.inic_mod == ModInic.INIC or self.curr_frame is None or self.left_line_inic_x is None :
            raise Exception("Greska u pipelineu.")

        prepocesed_frame = Preprocesing.preprocesing(self.curr_frame)

        changed_perspective = self.perspective_transform(prepocesed_frame)

        pair_of_lines = Histogram.make_pair_of_lines(changed_perspective)

        draw_sliding_window = changed_perspective.copy() # Da bi mi ostala i slika changed perspective bez sliding window
        left_fited_line, right_fited_line = self.call_sliding_window(draw_sliding_window, pair_of_lines)

        #self.update_radius_of_curvature(left_fited_line, right_fited_line, pair_of_lines)

        self.update_car_offset(left_fited_line,right_fited_line, pair_of_lines)

        plot = True
        if plot == True:
            cv2.imshow("original", self.curr_frame)
            cv2.imshow("changed perspective",changed_perspective)
            cv2.imshow("sliding window", draw_sliding_window)
            


        

