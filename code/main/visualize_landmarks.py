# Import the necessary modules
import cv2
import numpy as np

import sys
def show_points_as_contours(points):
    # Create an empty frame of size (640, 360)
    frame = np.zeros((360, 640), dtype=np.uint8)

    cv2.startWindowThread()
    cv2.namedWindow("Frame")
    # Loop over the frames in the points array
    for i in range(points.shape[0]):
        # Clear the frame
        frame[:] = 0
        
        # Convert the points to integers
        int_points = (points[i] * np.array([640, 360])).astype(np.int)
        
        # Draw the contours on the frame
        cv2.drawContours(frame, [int_points], -1, 255, 2)
        
        # Show the frame
        cv2.imshow("Frame", frame)
        
        # Wait for 50 ms
        cv2.waitKey(50)

    # Close the window
    cv2.destroyAllWindows()


# Test the function with a sample filename
if __name__ == "__main__":
    points = np.load(sys.argv[1], allow_pickle=True)
    show_points_as_contours(points)
