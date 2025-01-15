import cv2
import numpy as np

from frame import CameraCaptureFrame
from detector import RedDotDetector
from kalman_filter import KalmanFilter
from controller import Controller


video_path = "D:/workspace/object_tracing/2-D-Kalman-Filter-Trackor/video/red_dot.mp4"
cap = cv2.VideoCapture(video_path)
success, frame = cap.read()
display_name = "global tracker"
cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cv2.resizeWindow(display_name, 960, 720)
cv2.imshow(display_name, frame)

display_name_local = "local tracker"
cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

if success is not True:
    print("Read frame from {} failed.".format(video_path))
    exit(-1)

# Retrieve FPS from the video
fps = cap.get(cv2.CAP_PROP_FPS)
time_interval = 1 / fps
frame_tolerance = 10  # num of frames camera need to capture target

# object tracker 
detector = RedDotDetector()  

# estimate relative velocity 
kf = KalmanFilter(dt=time_interval, meas_std=0.01, acc_std=20)  

# control camera to keep target right in the middle
controller = Controller(dt=time_interval, detector=detector, kalman_filter=kf,
                        frame_tolerance=frame_tolerance)

frame_count = 0

while True:
    # cv.waitKey()
    # choose init camera frame
    frame_disp = frame.copy()
    cv2.putText(frame_disp, 'Select windows ROI and press ENTER', (20, 30), 
               cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1.5, (0, 0, 0), 1)
    
    # first frame correction(no prediction)
    x, y, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
    # x, y, w, h = 570, 226, 503, 355  # for debug
    v_init = np.array([0.0, 0.0], dtype=np.float32)
    cam_frame = CameraCaptureFrame(np.array([x, y, w, h]), 
                                   frame, 
                                   velocity=v_init)
    controller.detect(cam_frame) 

    # show local Kalman filter
    pre_x, pre_y = controller.kf.get_predcited_position()
    cor_x, cor_y = controller.kf.get_position()
    local_frame_disp = cam_frame.frame.copy()
    cx = cam_frame.center_x
    cy = cam_frame.center_y
    
    cv2.circle(local_frame_disp, (int(cx+pre_x), int(cy+pre_y)), 10,
               (255, 0, 0), -1)
    cv2.putText(local_frame_disp, 'P', (int(cx+pre_x+10), int(cy+pre_y)), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 0, 0), 1)
    cv2.circle(local_frame_disp, (int(cx+cor_x), int(cy+cor_y)), 10,
               (0, 255, 0), -1)
    cv2.putText(local_frame_disp, 'C', (int(cx+cor_x+10), int(cy+cor_y)), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (0, 255, 0), 1)
    cv2.putText(local_frame_disp, f'frame:{frame_count}', (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (0, 0, 255), 1)
    cv2.imshow(display_name_local, local_frame_disp)
    cv2.waitKey(1)

    break

while True:
    ret, frame = cap.read()
    if frame is None:
        break
    frame_count += 1

    frame_disp = frame.copy()

    # prediction
    pred_coor, velocity = controller.move_camera(cam_frame)  
    cam_frame = CameraCaptureFrame(pred_coor, frame,
                                   velocity=velocity)

    # detection and correction
    result = controller.detect(cam_frame) # x,y,w,h local

    # draw camera bbox(global)
    x_p, y_p, w_p, h_p = pred_coor
    cv2.rectangle(frame_disp, (int(x_p), int(y_p)), 
                  (int(x_p+w_p), int(y_p+h_p)),
                         (255, 0, 0), 5)
    cv2.putText(frame_disp, f'frame:{frame_count}', (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (0, 0, 255), 1)

    
    # draw target bbox(global)
    if result is not None:
        result = np.round(result).astype(np.int32)
        cv2.rectangle(frame_disp, 
                      (int(x_p+result[0]), int(y_p+result[1])), 
                      (int(x_p+result[0]+result[2]), int(y_p+result[1]+result[3])),
                            (0, 255, 0), 5)
    
    cv2.imshow(display_name, frame_disp)
    cv2.waitKey(1)

    # show local Kalman filter
    pre_x, pre_y = controller.kf.get_predcited_position()
    cor_x, cor_y = controller.kf.get_position()
    local_frame_disp = cam_frame.frame.copy()
    cx = cam_frame.center_x
    cy = cam_frame.center_y
    
    cv2.circle(local_frame_disp, (int(cx+pre_x), int(cy+pre_y)), 10,
               (255, 0, 0), -1)
    cv2.putText(local_frame_disp, 'P', (int(cx+pre_x+10), int(cy+pre_y)), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (255, 0, 0), 1)
    cv2.circle(local_frame_disp, (int(cx+cor_x), int(cy+cor_y)), 10,
               (0, 255, 0), -1)
    cv2.putText(local_frame_disp, 'C', (int(cx+cor_x+10), int(cy+cor_y)), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (0, 255, 0), 1)
    cv2.putText(local_frame_disp, f'frame:{frame_count}', (20, 20), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL,
                1, (0, 0, 255), 1)
    cv2.imshow(display_name_local, local_frame_disp)
    cv2.waitKey(1)




# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()