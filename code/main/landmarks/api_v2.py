import numpy as np
from typing import List, Tuple
import argparse
import dlib
import cv2
import os

WEIGHTS_PATH = "../assets/landmarks/shape_predictor_68_face_landmarks.dat"
# output_dir = '/storage3/demirtasm/data/tr_dataset/vis'

def extract_landmarks(video_path: str, face_detector: dlib.get_frontal_face_detector, landmark_predictor: dlib.shape_predictor,
                      visualize: bool = False):
    # Load the video
    video = cv2.VideoCapture(video_path)

    # Create a list to store the face landmarks
    landmarks = []

    frame_id = 0
    while True:
        # Read the next frame from the video
        ret, frame = video.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_detector(gray)
        if len(faces) == 0:
            landmarks.append(np.zeros((20, 2)))

        # For each face, extract the landmarks using the dlib predictor
        for face in faces[:1]:
            face_landmarks = landmark_predictor(gray, face)

            # Select only the landmarks for the lip area (48-68)
            lip_landmarks = np.array([(p.x, p.y) for p in face_landmarks.parts()[48:68]])

            # # # Normalize the landmarks using the bounding box around the lip area
            # left = min([x for (x, y) in lip_landmarks])
            # right = max([x for (x, y) in lip_landmarks])
            # top = min([y for (x, y) in lip_landmarks])
            # bottom = max([y for (x, y) in lip_landmarks])
            # w_scale = right - left + 1e-7
            # h_scale = bottom - top + 1e-7
            # min_ = np.array([left, top])
            # scale = np.array([w_scale, h_scale])
            min_, scale = 0., 1.
            normalized_landmarks = (lip_landmarks - min_) / scale

            # # Align the landmarks using the center of the lip area as the reference
            # cx = (left + right) / 2
            # cy = (top + bottom) / 2
            # aligned_landmarks = [(x - cx, y - cy) for (x, y) in normalized_landmarks]

            if visualize and frame_id % 50 == 0:
                # Draw the landmarks on the frame
                lip_contour = [(lip_landmarks[i][0], lip_landmarks[i][1]) for i in range(20)]
                frame = cv2.drawContours(frame, [np.array(lip_contour).reshape(-1, 2)], -1, (0, 0, 255), 1)

                norm = np.zeros((400, 400, 3), dtype=np.uint8)
                lip_contour = [(int(normalized_landmarks[i][0]*400), int(normalized_landmarks[i][1]*400)) for i in range(20)]
                norm = cv2.drawContours(norm, [np.array(lip_contour).reshape(-1, 2)], -1, (0, 0, 255), 1)

                # Save the frame to the output directory
                output_path = os.path.join(output_dir, f"{video_path.split('/')[-1]}_frame{frame_id:05d}.jpg")
                cv2.imwrite(output_path, frame)

                output_path = os.path.join(output_dir, f"{video_path.split('/')[-1]}_norm{frame_id:05d}.jpg")
                cv2.imwrite(output_path, norm)
                frame_id += 1

            # Append the aligned landmarks to the list
            landmarks.append(normalized_landmarks)

    # Convert the landmarks to NumPy array
    landmarks = np.stack(landmarks)

    return landmarks


def save_landmarks(landmarks, filename):
    # Save the landmarks array to a file using the NumPy save function
    np.save(filename, landmarks, allow_pickle=True)


def load_landmarks(filename):
    # Load the landmarks array from a file using the NumPy load function
    landmarks = np.load(filename, allow_pickle=True)

    return landmarks


def compare_landmarks(landmarks1, landmarks2, crop_length):
    # Crop the landmarks arrays to the specified length
    landmarks1 = landmarks1[:crop_length]
    landmarks2 = landmarks2[:crop_length]

    # Calculate the distance between the landmarks
    distance = 0
    for i in range(crop_length):
        distance += np.linalg.norm(landmarks1[i] - landmarks2[i])

    return distance / crop_length


def compare_videos(video_src: str, audio_src: str):
    # Create a face detector and a facial landmark predictor using dlib
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor(WEIGHTS_PATH)

    # Extract the landmarks for the first video
    landmarks1 = extract_landmarks(video_src, face_detector, landmark_predictor)

    # Extract the landmarks for the second video

    audio_landmarks_file = os.path.join('landmarks', audio_src + ".npy")
    if not os.path.exists(audio_landmarks_file):
        landmarks2 = extract_landmarks(audio_src, face_detector, landmark_predictor)
        save_landmarks(landmarks2, audio_landmarks_file)
    else:
        landmarks2 = load_landmarks(audio_landmarks_file)

    crop_length = min(len(landmarks1), len(landmarks2))

    # Compare the landmarks and return the result
    distance = compare_landmarks(landmarks1, landmarks2, crop_length)
    return distance, crop_length


def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("video_src", help="The path to the dubbed video")
    parser.add_argument("audio_src", help="The path to the audio source video")
    args = parser.parse_args()
    if not os.path.exists(args.video_src) or not os.path.exists(args.audio_src):
        print(args.video_src, -1, -1)

    distance, crop_length = compare_videos(args.video_src, args.audio_src)
    print(args.video_src, distance, crop_length)

if __name__ == "__main__":
    main()
