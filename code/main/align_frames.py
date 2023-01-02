'''
Warp face landmarks, crop face
'''
import dlib
import cv2
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt

WEIGHTS_PATH = "assets/landmarks/shape_predictor_68_face_landmarks.dat"

face_width = 120
face_height = 120
lip_width = 64
lip_height = 40

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor(WEIGHTS_PATH)

lk_params = dict(winSize  = (15, 15),
                 maxLevel = 2,
                 criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def get_landmarks(frame, visualize=False, normalize=False):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)
    if len(faces) == 0:
        raise Exception("No faces detected")

    # For each face, extract the landmarks using the dlib predictor
    for face in faces:
        landmarks = landmark_predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])

        top, bottom, left, right = face.top(), face.bottom(), face.left(), face.right()
        left = min([x for (x, y) in landmarks])
        right = max([x for (x, y) in landmarks])
        top = min([y for (x, y) in landmarks])
        bottom = max([y for (x, y) in landmarks])

        cropped_face = cv2.resize(gray[top:bottom, left:right], (120, 120))

        # Select only the landmarks for the lip area (48-68)
        lip_landmarks = landmarks[48:68]

        face_norm_factor = np.array([left, top]).reshape(1, 2)

        # Normalize the landmarks using the bounding box around the lip area
        left = min([x for (x, y) in lip_landmarks])
        right = max([x for (x, y) in lip_landmarks])
        top = min([y for (x, y) in lip_landmarks])
        bottom = max([y for (x, y) in lip_landmarks])
        lip_norm_factor = np.array([left, top]).reshape(1, 2)

        cropped_lip = gray[top:bottom, left:right]
        cropped_lip = cv2.resize(gray[top:bottom, left:right], (lip_width, lip_height))

        if normalize:
            w_scale = right - left + 1e-7
            h_scale = bottom - top + 1e-7
            min_ = np.array([left, top])
            scale = np.array([w_scale, h_scale])
        else:
            min_, scale = 0., 1.
        normalized_landmarks = (lip_landmarks - min_) / scale

        # Append the aligned landmarks to the list
        return {'face': cropped_face, 'face_offset': face_norm_factor,
                'lip': cropped_lip, 'lip_offset': lip_norm_factor,
                'lip_box': (bottom-top, right-left),
                'landmarks': lip_landmarks}



def zigzag(input):
    # from https://github.com/getsanjeev/compression-DCT/blob/master/zigzag.py
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = input.shape[0]
    hmax = input.shape[1]
    i = 0
    output = np.zeros(( vmax * hmax))
    while ((v < vmax) and (h < hmax)):
        if ((h + v) % 2) == 0:                 # going up
            if (v == vmin):
                output[i] = input[v, h]        # if we got to the first line
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                output[i] = input[v, h] 
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                output[i] = input[v, h] 
                v = v - 1
                h = h + 1
                i = i + 1
        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                output[i] = input[v, h] 
                h = h + 1
                i = i + 1        
            elif (h == hmin):                  # if we got to the first column
                output[i] = input[v, h] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1
            elif ((v < vmax -1) and (h > hmin)):     # all other cases
                output[i] = input[v, h] 
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            output[i] = input[v, h] 
            break
    return output




def calculate_cache(frame, prev_frame, cache, index):
    # Store the points in the cache if they do not already exist
    if index - 1 not in cache:
        frame_cache_prev = get_landmarks(prev_frame)
        cache[index - 1] = frame_cache_prev
    else:
        frame_cache_prev = cache[index - 1]

    if index not in cache:
        frame_cache = get_landmarks(frame)
        cache[index] = frame_cache
    else:
        frame_cache = cache[index]

    return frame_cache, frame_cache_prev


def calculate_lucas_kanade_optical_flow(prev_frame, curr_frame, prev_landmarks, curr_landmarks):
    # Calculate the optical flow using the Lucas-Kanade algorithm
    try:
        flow, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_landmarks.reshape(-1, 1, 2).astype(np.float32), None, winSize=(15,15),
                                        maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    except:
        print("LK Fail")
        flow = prev_landmarks[::]
    flow = flow.reshape(-1, 2)
    return flow


def calculate_motion_vectors(prev_landmarks, curr_landmarks):
    return curr_landmarks - prev_landmarks


def calculate_farneback_field(frame, prev_frame):
    # Use the Farneback algorithm to estimate the motion of the lips
    estimated_motion = cv2.calcOpticalFlowFarneback(prev_frame, frame, 
                                                    flow=None,
                                                    pyr_scale=0.5,
                                                    levels=3,
                                                    winsize=15,
                                                    iterations=3,
                                                    poly_n=5,
                                                    poly_sigma=1.2,
                                                    flags=0)
    return estimated_motion


def featurize_motion(estimated_motion):
    # Transform the motion vector component matrices using a 2D discrete cosine transform (DCT)
    Vx_transformed = cv2.dct(estimated_motion[:, :, 0])
    Vy_transformed = cv2.dct(estimated_motion[:, :, 1])

    # Combine the first 50 DCT coefficients of the zig-zag scan in both the x and y directions
    feature_vector = np.concatenate((zigzag(Vx_transformed)[:50], zigzag(Vy_transformed)[:50]))

    return feature_vector

def create_hsv_frame(motion):
    # Compute magnite and angle of 2D vector
    mag, ang = cv2.cartToPolar(motion[..., 0], motion[..., 1])
    # Create mask
    hsv_mask = np.zeros(motion.shape[:2]+(3,), dtype=np.float32)
    # Make image saturation to a maximum value
    hsv_mask[..., 1] = 255
    # Set image hue value according to the angle of optical flow
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    # Set value as per the normalized magnitude of optical flow
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to rgb
    face_motion_frame = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    return cv2.normalize(face_motion_frame, face_motion_frame, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def visualize_flow_field(flow):
    # Normalize the flow field
    flow /= np.abs(flow).max()

    # Convert to polar coordinates
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    # Create an image to draw on
    height, width = flow.shape[:2]
    image = np.zeros((height, width, 3), dtype=np.uint8)

    # Iterate over the flow field and draw the flow lines
    for y in range(0, height, 5):
        for x in range(0, width, 5):
            # Get the magnitude and angle of the flow at this point
            mag_val = mag[y, x]
            ang_val = ang[y, x]

            # Draw a line with length proportional to the magnitude and
            # direction determined by the angle
            dx = int(mag_val * np.cos(ang_val) * 10)
            dy = int(mag_val * np.sin(ang_val) * 10)
            image = cv2.line(image, (x, y), (x+dx, y+dy), 255, 1)
    return image

def draw_points_to_frame(points, frame_width=0, frame_height=0, frame=None):
    # Create an empty frame of size (640, 360)
    if frame is None:
        frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    # Loop over the frames in the points array
    for i in range(points.shape[0]):
        # Draw the contours on the frame
        frame = cv2.drawContours(frame, [points[i].reshape(1, 2)], -1, (0, 255, 0), 2)

    return frame

def main():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="input_video.mp4", help="the input video file")
    parser.add_argument("--output", default="output_video.mp4", help="the output video file")
    args = parser.parse_args()

    # Open the video file
    video = cv2.VideoCapture(args.input)

    # Read the first frame
    success, frame = video.read()

    # Set the previous frame to the first frame
    prev_frame = frame

    # Get the frame width and height
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Set the codec and create the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter("temp_video_1.mp4", fourcc, fps, (2*face_width, face_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out2 = cv2.VideoWriter("temp_video_2.mp4", fourcc, fps, (3*face_width, face_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out3 = cv2.VideoWriter("temp_video_3.mp4", fourcc, fps, (2*face_width, face_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out4 = cv2.VideoWriter("temp_video_4.mp4", fourcc, fps, (2*lip_width, lip_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out5 = cv2.VideoWriter("temp_video_5.mp4", fourcc, fps, (3*lip_width, lip_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out6 = cv2.VideoWriter("temp_video_6.mp4", fourcc, fps, (2*lip_width, lip_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out7 = cv2.VideoWriter("temp_video_7.mp4", fourcc, fps, (lip_width, lip_height), 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_lm = cv2.VideoWriter("temp_video_8.mp4", fourcc, fps, (lip_width, lip_height))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_hist= cv2.VideoWriter("temp_video_9.mp4", fourcc, fps, (600, 200))

    # Keep track of frame index
    i = 1

    # Create a cache for frame landmarks
    video_cache = {}

    # Iterate through the frames in the video
    while success:
        # Align the current frame with the previous frame
        frame_cache, frame_cache_prev = calculate_cache(frame, prev_frame,
                                                        cache=video_cache,
                                                        index=i)
        i += 1

        # Set the previous frame to the current frame
        prev_frame = frame
        # Read the next frame
        success, frame = video.read()

        flow = calculate_lucas_kanade_optical_flow(frame_cache_prev['face'],
                                                frame_cache['face'],
                                                frame_cache_prev['landmarks'] - frame_cache_prev['face_offset'],
                                                frame_cache['landmarks'] - frame_cache['face_offset'],
                                                )

        # Extract motion field from face crop
        face_motion = calculate_farneback_field(frame_cache['face'], frame_cache_prev['face'])
        face_motion_frame = create_hsv_frame(face_motion)
        out1.write(np.hstack([face_motion_frame, np.repeat(frame_cache['face'].reshape(120, 120, 1), 3, axis=2)]))

        Vx, Vy = face_motion[..., 0], face_motion[..., 1]
        norm = lambda t: np.clip(255 * (t - t.min()) / (t.max() - t.min()), 0, 255).astype(np.uint8).reshape(120, 120, 1)
        Vxn = np.repeat(norm(Vx).reshape(120, 120, 1), 3, axis=2)
        Vyn = np.repeat(norm(Vy).reshape(120, 120, 1), 3, axis=2)
        out2.write((np.hstack([Vxn, Vyn, np.repeat(frame_cache['face'].reshape(120, 120, 1), 3, axis=2)])))
        
        face_map = visualize_flow_field(face_motion)
        out3.write(np.hstack([face_map, np.repeat(frame_cache['face'].reshape(120, 120, 1), 3, axis=2)]))


        # Extract motion field from lip crop
        lip_motion = calculate_farneback_field(frame_cache['lip'], frame_cache_prev['lip'])
        lip_motion_frame = create_hsv_frame(lip_motion)
        out4.write(np.hstack([lip_motion_frame, np.repeat(frame_cache['lip'].reshape(lip_height, lip_width, 1), 3, axis=2)]))

        Vx, Vy = lip_motion[..., 0], lip_motion[..., 1]
        norm = lambda t: np.clip(255 * (t - t.min()) / (t.max() - t.min()), 0, 255).astype(np.uint8).reshape(lip_height, lip_width, 1)
        Vxn = np.repeat(norm(Vx).reshape(lip_height, lip_width, 1), 3, axis=2)
        Vyn = np.repeat(norm(Vy).reshape(lip_height, lip_width, 1), 3, axis=2)
        out5.write((np.hstack([Vxn, Vyn, np.repeat(frame_cache['lip'].reshape(lip_height, lip_width, 1), 3, axis=2)])))

        lip_map = visualize_flow_field(lip_motion)
        out6.write(np.hstack([lip_map, np.repeat(frame_cache['lip'].reshape(lip_height, lip_width, 1), 3, axis=2)]))


        # Extract vectors for lip landmarks
        vectors = calculate_motion_vectors(frame_cache_prev['landmarks'],
                                frame_cache['landmarks'],
                                )

        # Iterate over the points and vectors
        img = np.zeros(frame_cache_prev['lip_box'], dtype=np.uint8)
        for point, vector in zip(frame_cache_prev['landmarks']-frame_cache_prev['lip_offset'], vectors):
            # Draw an arrow on the image to represent the vector
            img = cv2.arrowedLine(img, tuple(point), tuple(point + vector), 255, 1)
        img = cv2.resize(img, (lip_width, lip_height))
        out7.write(img)

        # Extract motion features from lip area
        feature_vector = featurize_motion(lip_motion)
        f = plt.figure(figsize=(3,1))
        plt.bar(np.arange(100), feature_vector)
        plt.ylim(-20, 20)
        plt.subplots_adjust(top=0.925, 
                            bottom=0.25, 
                            left=0.15, 
                            right=0.90, 
                            hspace=0.01, 
                            wspace=0.01)

        f.canvas.draw()
        f_arr = np.array(f.canvas.renderer._renderer)
        plt.close()
        f_arr = cv2.resize(f_arr,(600,200))
        bgr = cv2.cvtColor(f_arr, cv2.COLOR_RGBA2BGR)
        out_hist.write(bgr)

        landmark_frame = draw_points_to_frame(frame_cache['landmarks'] - frame_cache['lip_offset'], frame=frame_cache['lip'])
        out_lm.write(np.repeat(landmark_frame.reshape(lip_height, lip_width, 1), 3, axis=2))

    # Release the video capture and video writer objects
    video.release()
    out1.release()
    out2.release()
    out3.release()
    out4.release()
    out5.release()
    out6.release()
    out7.release()
    out_lm.release()
    out_hist.release()

    for i in range(1, 10):
        # Extract the audio from the input video and save it to a temporary file
        audio_temp_file = f"temp_audio_{i}.aac"
        video_temp_file = f"temp_video_{i}.mp4"
        command = f"ffmpeg -y -i {args.input} -vn -acodec copy {audio_temp_file} -hide_banner -loglevel error"
        subprocess.run(command, shell=True)

        # Combine the output video and the extracted audio into a single file
        command = f"ffmpeg -y -i {video_temp_file} -i {audio_temp_file} -c copy -map 0:v:0 -map 1:a:0 method{i}_{args.output} -hide_banner -loglevel error"
        subprocess.run(command, shell=True)

        command = f"rm {video_temp_file} {audio_temp_file}"
        subprocess.run(command, shell=True)


if __name__ == "__main__":
    main()



