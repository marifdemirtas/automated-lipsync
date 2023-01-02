'''
In a data folder,
1- find the longest videos of each speaker
2- Dub each video in the folder using the longest video of each speaker
3- save to OUT_PATH

Reads config from config.py
'''

import argparse
import time

import numpy as np
from PIL import Image
from tqdm import tqdm
import glob

from media import VideoCamera
from config import Config

import os
import cv2
import pandas as pd
import random
import pickle
import subprocess


DATA_PATH = "/storage3/demirtasm/data/tr_dataset/data/processed/vid"
OUT_PATH = "/storage3/demirtasm/code/lipsync/results/tr_model"

def dataset(path):
    '''
    Get files inside a data folder
    '''
    return glob.glob(os.path.join(path, '*'))

def speaker(file):
    '''
    Get the speaker from a video title
    '''
    return file.split('/')[-1].rstrip('.mp4').split('_')[1]

def length(filename):
    '''
    Get the length of a video in seconds
    '''
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                             "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    return float(result.stdout)


if __name__ == "__main__":

    max_lens = {}
    max_names = {}

    app_config = vars(Config)

    # Find longest video of each speaker
    for file in tqdm(dataset(DATA_PATH), desc='Dataset traversal', leave=False, ascii=True):
        if max_lens.get(speaker(file), 0) < length(file):
            max_names[speaker(file)] = file
            max_lens[speaker(file)] = length(file)
            
    # Traverse videos in dataset
    for face_src in tqdm(dataset(DATA_PATH), desc='Face sources', leave=False, ascii=True):
        for audio_src in tqdm(max_names.values(), ascii=True, leave=True, desc='Audio sources for given head'):
            if audio_src == face_src:
                continue
            with VideoCamera(src=face_src, audio=audio_src, app_config=app_config, loop=False) as cam:
                cam.i = 0

                # Run dubbing
                name = f"{face_src.split('/')[-1]}_TO_{audio_src.split('/')[-1]}"
                frames = cam.calculate_frames()
                cam.save_frames_to_video(frames, os.path.join(OUT_PATH, f"{name}.mp4"))
