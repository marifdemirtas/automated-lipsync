import dlib
import numpy as np
from tqdm import tqdm
import cv2

class LandmarkDetector():
    def __init__(self,
                 model_path="assets/landmarks/D40_C3.h5",
                 predictor_path="assets/landmarks/shape_predictor_68_face_landmarks.dat"):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

        self.wsize = 0.05 # .04
        self.hsize = 0.0125 # self.wsize
        self.ctxWin = 3
        self.num_features_Y = 136
        self.delay = 1
        self.num_frames = 75


    def detect_landmarks(self, frame):
        points = np.zeros((20, 2), dtype=np.float32)
        try:
            dets = self.detector(frame, 1)
        except Exception as e:
            print("Exception in detect_landmarks:", e)
            return points
        if len(dets) != 1:
            # print('FACE DETECTION FAILED!!')
            return points

        for k, d in enumerate(dets):
            h, w, _ = frame.shape
            shape = self.predictor(frame, d)

            for i in range(48, 68):
                points[i-48, 0] = shape.part(i).x / h
                points[i-48, 1] = shape.part(i).y / w

        return points

    def draw_landmarks(self, frame, landmarks):
        for row in landmarks.astype(np.int32):
            frame = cv2.circle(frame, (row[0], row[1]), 3, (0, 0, 255), -1)
        return frame

    def align_landmarks(self, landmarks):
        aligned = np.zeros_like(landmarks)
        for i, frame in enumerate(landmarks):
            center = frame[0]
            angle = np.arctan((frame[6, 1] - frame[0, 1]) / (frame[6, 0] - frame[0, 0] + 1e-7)) * 57.29
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.)
            aligned[i] = np.matmul(rot_mat, np.c_[frame, np.ones(20)].T).T
        return aligned
    
    def normalize_landmarks(self, landmarks):
        normalized = np.zeros_like(landmarks)
        for i, frame in enumerate(landmarks):
            w_scale = frame[6,0] - frame[0,0] + 1e-7
            h_scale = frame[9,1] - frame[3,1] + 1e-7
            min = np.array([frame[0, 0], frame[3, 1]])
            scale = np.array([w_scale, h_scale])
            normalized[i] = (frame - min) / scale
        return normalized
