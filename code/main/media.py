import cv2
import numpy as np
import torch
import time
import subprocess, platform
from tqdm import tqdm

from wav2lip.api import melspectrogram, load_wav
from wav2lip.api import ModelWrapper as Wav2Lip
from landmarks.api import LandmarkDetector
#from talking_face.api import ModelWrapper as PCAVS

ARCHS = {
    'wav2lip': Wav2Lip,
    # 'pcavs': PCAVS
}

#### VIDEO ####

TARGET_FPS = 25

class ImageCapture():
    def __init__(self, src):
        self.frame = cv2.imread(src)
    
    def release(self):
        pass

    def read(self):
        return 0, self.frame.copy()

class VideoCamera(object):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.video.release()
        print(f"VideoCam finished")

    def __init__(self, src: str, audio: str,
                 loop: bool = True,
                 app_config={}):
        self.audio_path = audio

        Model = ARCHS[app_config.get('MODEL_ARCH', 'wav2lip')]
        # LandmarkDetector

        if src == None:
            self.video = cv2.VideoCapture(0)
        elif src.endswith('mp4'):
            self.video = cv2.VideoCapture(src)
        else:
            self.video = ImageCapture(src)

        opts = {
            'model': app_config.get('MODEL_TYPE', 'lq'),
            'compiled': app_config.get('COMPILE', True),
            'face': app_config.get('FACE', 'fast'),
            'image_size': app_config.get('IMAGE_SIZE', 96),
        }
        
        self.model = Model(opts)
        
        self.mel = extract_mel(audio,
                               app_config.get('STEP_SIZE', 16),
                               app_config.get('SR', 16000))


        self.landmark_detector = LandmarkDetector()

        self.i = -10
        self.raw_frames = []

        self.wait = 1. / TARGET_FPS
        self.loop = loop
        self.result = None
        if app_config.get('RUNTIME_MODE') == 'offline':
            self.write_video_async("/Users/mehmetarifdemirtas/Documents/Bitirme/main/static/tmp.mp4")

    def __del__(self):
        self.video.release()        

    def get_frame(self):
        ret, frame = self.video.read()
        if ret is False:
            self.i = -1

        if self.i >= 0:
            data = {'img': frame, 'mel': self.mel[self.i]}
            frame = self.model.infer(data)
            if frame is not None:
                self.i += 1
                if self.i >= len(self.mel) and self.loop:
                    self.i = 0
            else:
                frame = np.zeros(self.frame[-1].shape).astype(np.uint8)
            return frame
        else:
            return False

    def encoded_frame(self):
        st = time.time()
        frame = self.get_frame()
        ret, jpeg = cv2.imencode('.jpg', frame)
        et = time.time()
        print("time:", et-st)
        time.sleep(max(self.wait-(et-st), 0))
        return jpeg.tobytes()

    def calculate_frames(self):
        frames = [self.get_frame() for _ in tqdm(range(len(self.mel)), ascii=True, desc='Face dubbing frames')]
        return list(filter(lambda v: not isinstance(v, bool), frames))

    def save_frames_to_video(self, frames, filename):
        frame_h, frame_w = frames[0].shape[:-1]
        out = cv2.VideoWriter('/tmp/result.avi', 
                              cv2.VideoWriter_fourcc(*'DIVX'), TARGET_FPS, (frame_w, frame_h))
        for frame in frames:
            try:
                out.write(frame)
            except:
                import pdb
                pdb.set_trace()
        out.release()
        command = 'ffmpeg -y -i {} -vn -acodec copy /tmp/outsound.aac -hide_banner -loglevel error'.format(self.audio_path)
        subprocess.call(command, shell=platform.system() != 'Windows')

        command = 'ffmpeg -y -i /tmp/outsound.aac -i /tmp/result.avi -strict -2 -q:v 1 -shortest {} -hide_banner -loglevel error'.format(filename)
        subprocess.call(command, shell=platform.system() != 'Windows')
        subprocess.call('rm /tmp/result.avi', shell=platform.system() != 'Windows')
        self.result = filename

    def write_video_async(self, out_path):
        self.i = 0
        st = time.time()
        frames = self.calculate_frames()
        et = time.time()
        print("time:", et-st)
        self.save_frames_to_video(frames, out_path)

    def __del__(self):
        self.video.release()

time_start = time.time()
times = []

def generate_video(camera):
    while True:
        global time_start
        times.append(time.time() - time_start)
        time_start = time.time()
        frame = camera.encoded_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


#### AUDIO ####

def extract_mel(src: str, step_size: int, sr: int):
    mel_idx_multiplier = 80./TARGET_FPS 
    mel_step_size = step_size
    # 16 for wav2lip, 20 for talkingface

    mel = melspectrogram(load_wav(src, sr))
    mel_chunks = []
    i = 0
    while 1:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
        i += 1
    return mel_chunks


def generate_audio(src: str):
    chunk_size = 256
    with open(src, "rb") as fwav:
        data = fwav.read(chunk_size)
        while data:
            yield data
            data = fwav.read(chunk_size)
            if len(data) < chunk_size:
                fwav.seek(0)
                data += fwav.read(chunk_size - len(data))
