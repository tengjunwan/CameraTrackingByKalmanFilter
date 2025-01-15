import numpy as np
import cv2


class CameraCaptureFrame():

    def __init__(self, global_coor ,global_frame, velocity):
        self.global_coor = global_coor

        self.width = self.global_coor[2]
        self.hight = self.global_coor[3]
        self.global_center_x = self.global_coor[0] + 0.5 *self.global_coor[2]
        self.global_center_y = self.global_coor[1] + 0.5 *self.global_coor[3]
        self.center_x = 0.5 *self.global_coor[2]
        self.center_y = 0.5 *self.global_coor[3]
        # only keep local frame 
        
        p1_x, p1_y, w, h = np.round(self.global_coor).astype(np.int32)
        p2_x = p1_x + w
        p2_y = p1_y + h
        # out of frame
        pad_x1 = abs(min(0, p1_x))
        pad_y1 = abs(min(0, p1_y))
        pad_x2 = max(0, p2_x - global_frame.shape[1])
        pad_y2 = max(0, p2_y - global_frame.shape[0])

        self.frame = global_frame[
            max(0, p1_y): min(p2_y, global_frame.shape[0]),
            max(0, p1_x):min(p2_x, global_frame.shape[1])
            :].copy()
        self.frame = cv2.copyMakeBorder(self.frame, 
                                        pad_y1, pad_y2, pad_x1, pad_x2, 
                                        cv2.BORDER_CONSTANT, None, (0,0,0))
        if len(self.frame.flatten()) == 0:
            raise ValueError("here")
        self.velocity = velocity
        
        

        
        
        



    