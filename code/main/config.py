import os

class Config:

    ROOT_FOLDER = ""
    DEFAULT_AUDIO = os.path.join(ROOT_FOLDER, "files/audio.wav")

    UPLOAD_FOLDER = os.path.join(ROOT_FOLDER, "main/files_upload")

    ALLOWED_EXTENSIONS = {
    'audio': ['wav', 'mp4'],
    'image': ['png', 'jpg', 'jpg'],
    'video': ['mp4']
    }

    SECRET_KEY = "abc"

    MODEL_TYPE = 'lq' # ['lq', 'hq']
    MODEL_ARCH = 'wav2lip' # ['wav2lip', 'pcavs']
    COMPILE = False
    FACE = 'fast' # ['fast', 's3fd']
    IMAGE_SIZE = 96 # 96 for wav2lip, 224 for pcavs

    STEP_SIZE = 16 # 16 for wav2lip, 20 for pcavs
    SR = 16000

    RUNTIME_MODE = None