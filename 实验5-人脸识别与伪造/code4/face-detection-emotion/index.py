import threading
import cv2
import os
import json
import dlib

from deepface import DeepFace
 
# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


class OpcvCapture(threading.Thread):
    def __init__(self, win_name, cam_name):
        super().__init__()
        self.cam_name = cam_name
        self.win_name = win_name
        # self.frame = np.zeros([400, 400, 3], np.uint8)
 
    def run(self):
        #capture = cv2.VideoCapture(self.cam_name)
        capture = cv2.VideoCapture("birthday.mp4")
        
        # TODO
 
 
if __name__ == "__main__":
    camera1 = OpcvCapture("camera1", 0)
    camera1.start()