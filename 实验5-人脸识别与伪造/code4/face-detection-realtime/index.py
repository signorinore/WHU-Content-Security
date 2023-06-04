import threading
import cv2
import os
import json
import dlib
 
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name
 
    def run(self):
        #capture = cv2.VideoCapture(self.cam_name)
        capture = cv2.VideoCapture("birthday.mp4")
        
        while (True):
            # 获取一帧
            ret, frame = capture.read()
            
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = detector(frame)
        
  
if __name__ == "__main__":
    camera1 = OpcvCapture("Face", 0)
    camera1.start()
