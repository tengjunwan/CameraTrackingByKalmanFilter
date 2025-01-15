import numpy as np
import cv2


class RedDotDetector():
    """simple red dot detector for exp"""

    def __init__(self):
        self.lower_red1 = np.array([0, 50, 50])     # First range for red (hue around 0°)
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])   # Second range for red (hue around 180°)
        self.upper_red2 = np.array([180, 255, 255])
        self.smallest_area = 5
        

    def detect(self, frame):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create masks for both red ranges and combine them
        mask1 = cv2.inRange(hsv_image, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv_image, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)

        # Find contours from the mask
        contours, _ = cv2.findContours(mask, 
                                       cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around detected red dots
        detect = False
        for contour in contours:
            if cv2.contourArea(contour) > self.smallest_area:  
                x, y, w, h = cv2.boundingRect(contour)
                detect = True
                break

        if detect:
             return np.array([x, y, w, h], dtype=np.float32)
        else:
             return None
        

if __name__ == "__main__":
    det = RedDotDetector()
    img_path = "D:/workspace/object_tracing/2-D-Kalman-Filter-Trackor/img/1.jpg"
    img = cv2.imread(img_path)
    result = det.detect(img)
    if result is not None:
        x, y, w, h = np.round(result).astype(np.int32)
        img = cv2.rectangle(img, (x, y), (x + w, y + h), 
                      (0, 255, 0), 2)  # Green bounding box

    # Show the results
    cv2.imshow('Detected Red Dot', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()