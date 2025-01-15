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
        g_x1, g_y1, w, h = np.round(self.global_coor).astype(np.int32)
        g_x2 = g_x1 + w
        g_y2 = g_y1 + h
        overlap = get_overlap((g_x1, g_y1, g_x2, g_y2), 
                              (0, 0, global_frame.shape[1], global_frame.shape[0]))
        self.frame = np.zeros((h, w, 3), dtype=np.uint8)
        if overlap is not None:
            o_x1, o_y1, o_x2, o_y2 = overlap
            # convert overlap coor to local frame
            l_x1 = o_x1 - g_x1
            l_y1 = o_y1 - g_y1
            l_x2 = o_x2 - g_x1
            l_y2 = o_y2 - g_y1
            # copy from global frame
            self.frame[l_y1: l_y2, l_x1: l_x2 , :] = global_frame[o_y1: o_y2,
                                                                  o_x1: o_x2,
                                                                  :]

        if len(self.frame.flatten()) == 0:
            raise ValueError("here")
        self.velocity = velocity
        
        
def get_overlap(box1, box2):
    """
    Calculate the overlap (intersection) of two boxes.

    Parameters:
        box1: tuple (x1, y1, x2, y2) for the first box
        box2: tuple (x1, y1, x2, y2) for the second box

    Returns:
        tuple (x1, y1, x2, y2) for the overlapping box if there is an overlap,
        otherwise None.
    """
    # Coordinates of the intersection box
    x1 = max(box1[0], box2[0])  # Max of top-left x-coordinates
    y1 = max(box1[1], box2[1])  # Max of top-left y-coordinates
    x2 = min(box1[2], box2[2])  # Min of bottom-right x-coordinates
    y2 = min(box1[3], box2[3])  # Min of bottom-right y-coordinates

    # Check if the boxes overlap
    if x1 < x2 and y1 < y2:
        return (x1, y1, x2, y2)  # Return the coordinates of the overlap
    else:
        return None  # No overlap
        
        
        



    