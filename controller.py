import numpy as np


class Controller:

    def __init__(self, dt, detector, kalman_filter, frame_tolerance=30):
        self.dt = dt
        self.delta_t_for_acc = dt * frame_tolerance
        self.detector = detector
        self.kf = kalman_filter
        self.last_acc = 0.0

    def _cal_acc(self, rela_p, rela_v):
        # rela_p = [x, y], rela_v = [vx, vy]
        acc = (2 / self.delta_t_for_acc**2) * rela_p \
            + (2 / self.delta_t_for_acc) * rela_v
        # acc = (2 / self.dt**2) * rela_p \
        #     + (2 / self.dt) * rela_v
        return acc  # [ax, ay]
    
    def detect(self, cam_frame):
        result = self.detector.detect(cam_frame.frame)  # x, y, w, h, local detect
        if result is not None:
            # convey to relative position w.r.t center of frame
            cx = result[0] + 0.5 * result[2]
            cy = result[1] + 0.5 * result[3]
            rel_x = cx - 0.5 * cam_frame.width
            rel_y = cy - 0.5 * cam_frame.hight
            
            # correct prediction of Kalman Filter by detection
            self.kf.correct(np.array([rel_x, rel_y]))
        else:
            self.kf.correct(None)

        return result
      
    def move_camera(self, cam_frame):
        # calculate accelerations needed 
        acc = self._cal_acc(self.kf.get_position(),
                            self.kf.get_velocity())
        self.last_acc = acc
        self.kf.predict(acc)
        # udpate global position, velocity of camera frame
        new_velocity = cam_frame.velocity + acc * self.dt
        new_global_coor = np.zeros(4)  # x, y , w, h
        new_global_coor[:2] = cam_frame.global_coor[:2] \
            + cam_frame.velocity * self.dt + 0.5 * acc * self.dt**2
        new_global_coor[2:] = cam_frame.global_coor[2:]


        return new_global_coor, new_velocity





        

    
