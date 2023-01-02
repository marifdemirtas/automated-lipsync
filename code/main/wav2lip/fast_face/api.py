import numpy as np
import cv2
import os

from .vision.ssd.config.fd_config import define_img_size
from .vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from .vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor


class FaceDetector():
    def __init__(self, net_type='slim'):
        self.threshold = 0.6
        self.candidate_size = 1500
        self.input_size = 640
        class_size = 2

        define_img_size(self.input_size)
        
        if net_type == 'slim':
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained/version-slim-320.pth")
            # model_path = "models/pretrained/version-slim-640.pth"
            net = create_mb_tiny_fd(class_size, is_test=True, device='cpu')
            net.load(model_path)
            self.predictor = create_mb_tiny_fd_predictor(net, candidate_size=self.candidate_size, device='cpu')
        elif net_type == 'RFB':
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained/version-RFB-320.pth")
            # model_path = "models/pretrained/version-RFB-640.pth"
            net = create_Mb_Tiny_RFB_fd(class_size, is_test=True, device='cpu')
            net.load(model_path)
            self.predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=self.candidate_size, device='cpu')


    def detect(self, frame):
        boxes, labels, probs = self.predictor.predict(frame, self.candidate_size / 2, self.threshold)
        if not len(probs) == 0 and probs.max() > 0.5:
            box = [int(i) for i in boxes.numpy()[np.argmax(probs)]]
        else:
            box = [0, 0, 0, 0]

        return box

    def get_detections_for_batch(self, batch):
        return [self.detect(image) for image in batch]