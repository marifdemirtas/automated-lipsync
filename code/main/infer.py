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
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--face_src", help="The path to the face src")
    parser.add_argument("--audio_src", help="The path to the audio src")
    parser.add_argument("--src_file", help="The path to the csv file with sources")
    parser.add_argument("--result_file", default=None, help="The path to the result file")
    args = parser.parse_args()

    if args.face_src and args.audio_src:
        sources = [(args.face_src, args.audio_src)]
    elif args.src_file:
        with open(args.src_file, "r") as f:
            # Create a CSV reader
            reader = csv.reader(f)

            # Create two empty lists to store the data
            face_sources = []
            audio_sources = []

            # Iterate over the rows of the CSV file
            for row in reader:
                # Append the first and second elements of the row to the corresponding lists
                face_sources.append(row[0])
                audio_sources.append(row[1])
        sources = zip(face_sources, audio_sources)
    else:
        parser.error("Error: either face_src and audio_src OR src_file are required.")


    app_config = vars(Config)
    for face_src, audio_src in sources:
        with VideoCamera(src=face_src, audio=audio_src, app_config=app_config, loop=False) as cam:
            cam.i = 0
            # Run dubbing
            frames = cam.calculate_frames()
            name = args.result_file or f"{face_src.split('/')[-1]}_TO_{audio_src.split('/')[-1]}"
            cam.save_frames_to_video(frames, name)
