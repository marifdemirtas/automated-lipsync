'''
Evaluate videos created by benchmark.py
'''
import os
import argparse
import pandas as pd
import dlib
from tqdm import tqdm
from landmarks.api_v2 import extract_landmarks, compare_landmarks, \
    save_landmarks, load_landmarks

def extract_file_names(filename):
    # Split the file name by the delimiter
    parts = filename[:-4].split("_TO_")

    # The video source file is the first part before the delimiter
    video_src = parts[0]

    # The audio source file is the second part after the delimiter
    audio_src = parts[1]

    # The dubbed file is the second part with the ".mp4" extension removed
    dubbed_src = filename

    return video_src, audio_src, dubbed_src


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir1", required=True, help="The path to the directory containing the dubbed videos")
    parser.add_argument("--dir2", required=True, help="The path to the directory containing the source videos")
    parser.add_argument("--landmark_dir", required=True, help="The path to the directory containing landmark.npy files")
    args = parser.parse_args()

    # Create a face detector and a facial landmark predictor using dlib
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("assets/landmarks/shape_predictor_68_face_landmarks.dat")

    # Try to load the dataframe from "results.csv"
    try:
        results = pd.read_csv("results.csv")
    except FileNotFoundError:
        results = pd.DataFrame(columns=["dubbed_src", "video_src", "audio_src", "video_dubbed_distance", "audio_dubbed_distance", "video_audio_distance", "crop_length"])

    # Iterate over the files in the first directory
    for filename in tqdm(os.listdir(args.dir1), ascii=True, desc='Iterating over dubbed files'):
        # Extract the source and dubbed file names from the file name
        video_src, audio_src, dubbed_src = extract_file_names(filename)
        if results["dubbed_src"].isin([dubbed_src]).any():
            continue

        # Check if the video source and audio source files exist in the second directory
        video_path = os.path.join(args.dir2, video_src)
        audio_path = os.path.join(args.dir2, audio_src)
        if not os.path.exists(video_path) or not os.path.exists(audio_path):
            continue

        # Extract the landmarks from the video source and audio source files
        video_landmarks_file = os.path.join(args.landmark_dir, video_src + ".npy")
        audio_landmarks_file = os.path.join(args.landmark_dir, audio_src + ".npy")
        if not os.path.exists(video_landmarks_file):
            video_landmarks = extract_landmarks(video_path, face_detector, landmark_predictor)
            save_landmarks(video_landmarks, video_landmarks_file)
        else:
            video_landmarks = load_landmarks(video_landmarks_file)

        if not os.path.exists(audio_landmarks_file):
            audio_landmarks = extract_landmarks(audio_path, face_detector, landmark_predictor)
            save_landmarks(audio_landmarks, audio_landmarks_file)
        else:
            audio_landmarks = load_landmarks(audio_landmarks_file)

        # Check if the dubbed video file exists in the second directory
        dubbed_path = os.path.join(args.dir1, dubbed_src)
        if not os.path.exists(dubbed_path):
            print("NOT EXISTS:", dubbed_path)
            continue

        # Extract the landmarks from the dubbed video file
        dubbed_landmarks = extract_landmarks(dubbed_path, face_detector, landmark_predictor, visualize=True)

        crop_length = min([video_landmarks.shape[0], audio_landmarks.shape[0], dubbed_landmarks.shape[0]])
        if crop_length == 0:
            continue

        # Calculate the distances between the video source and audio source files
        video_audio_distance = compare_landmarks(video_landmarks, audio_landmarks, crop_length)

        # Calculate the distances between the dubbed video file and the video source and audio source files
        video_dubbed_distance = compare_landmarks(video_landmarks, dubbed_landmarks, crop_length)
        audio_dubbed_distance = compare_landmarks(audio_landmarks, dubbed_landmarks, crop_length)

        # Add the results to the dataframe
        results.loc[len(results)] = [dubbed_src, video_src, audio_src, video_dubbed_distance, audio_dubbed_distance, video_audio_distance, crop_length]

        # Save the dataframe to a CSV file
        results.to_csv("results.csv", index=False)

if __name__ == "__main__":
    main()