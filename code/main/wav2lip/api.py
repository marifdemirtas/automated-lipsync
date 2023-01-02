import torch
#from openvino.runtime import Core

from .model import Wav2Lip
from .audio import melspectrogram, load_wav
from .face_detection import FaceAlignment, LandmarksType
from .fast_face import FaceDetector
import os
import numpy as np
import cv2

def check_box(shape, box):
    x0, y0, x1, y1, = box
    y0 = max(y0, 0)
    x0 = max(x0, 0)
    y1 = min(y1, shape[0])
    x1 = min(x1, shape[1])

    box = [x0, y0, x1, y1]
    return box

def square_box(box):
    x0, y0, x1, y1, = box
    vertical, horizontal = y1-y0, x1-x0
    import pdb
    if vertical > horizontal:
        diff = (vertical-horizontal) // 2
        x0 -= diff
        x1 += diff
    else:
        diff = (horizontal-vertical) // 2
        y0 -= diff
        y1 += diff

    box = [x0, y0, x1, y1]
    return box

class ModelWrapper:
    def __init__(self, opts):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = load_model(opts['model'], compile=opts['compiled'], device=self.device)

        self.image_size = opts['image_size']

        if opts['face'] == 'fast':
            self.detector = FaceDetector()
        else:
            class DetectorWrapper():
                def __init__(self):
                    self.det = FaceAlignment(LandmarksType._2D, 
                                             flip_input=False, device='cpu')

                def detect(self, frame):
                    return self.det.get_detections_for_batch(np.expand_dims(frame, 0))[0]
            self.detector = DetectorWrapper()

    def infer(self, data):
        frame, spectrogram = data['img'].copy(), data['mel']
 
        box = self.detector.detect(frame) # TODO check box coordinates for frame borders
        box = check_box(frame.shape, box)
        if sum(box) == 0:
            return None
        else:
            face = cv2.resize(frame[box[1]:box[3], box[0]:box[2]], (self.image_size, self.image_size))
            image = np.asarray([face])
            mask = image.copy()
            mask[:, self.image_size//2:] = 0
            image = np.concatenate((mask, image), axis=3) / 255.

            mel = np.asarray([spectrogram])
            mel = np.reshape(mel, [len(mel), mel.shape[1], mel.shape[2], 1])
        
            with torch.no_grad():
                image_tensor = torch.FloatTensor(np.transpose(image, (0, 3, 1, 2))).to(self.device)
                mel_tensor = torch.FloatTensor(np.transpose(mel, (0, 3, 1, 2))).to(self.device)
                pred = self.model([mel_tensor, image_tensor])
                del mel_tensor
                del image_tensor

            pred = pred[0].transpose(1, 2, 0) * 255.
            
            x1, y1, x2, y2 = box
            pred = cv2.resize(pred.astype(np.uint8), (x2 - x1, y2 - y1))
            frame[y1:y2, x1:x2] = pred
            return frame


MODELS = {
    'lq': 'wav2lip_gan_lq',
    'hq': 'wav2lip_gan_hq',
    'tr': 'wav2lip_tr',
}

def _load(checkpoint_path):
    checkpoint = torch.load(checkpoint_path,
                            map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(model_name='lq', compile=True, device=torch.device('cpu')):

    if compile:
        ie = Core()
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/compiled', MODELS[model_name], "model.xml")
        model_ir = ie.read_model(model=path)
        compiled_model = ie.compile_model(model=model_ir, device_name="CPU")
        output_layer = compiled_model.output(0)
        model = lambda inp: compiled_model(inp)[output_layer]
    else:
        model_name = MODELS[model_name]
        path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data', f"{model_name}.pth")

        model = Wav2Lip()

        checkpoint = _load(path)
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)

        model_ = model.to(device).eval()
        model = lambda inp: model_(inp).cpu().numpy()
    return model

def export_model():
    model = load_model()
    image = torch.randn(1, 6, 96, 96)
    mel = torch.randn(1, 1, 80, 16)

    torch.onnx.export(model,
                      [mel, image],
                      "model.onnx",
                      opset_version=11)

if __name__ == "__main__":
    export_model()