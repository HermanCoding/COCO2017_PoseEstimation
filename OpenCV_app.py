import cv2
import numpy as np
from ultralytics import YOLO

# Pretrained model from Ultralytics
model = YOLO("yolov8n-pose.pt")


# Calculate any angel using three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def open_webcam():
    # Capture webcam
    cap = cv2.VideoCapture(0)  # 0 is default webcam.

    # Original frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create a window with the original frame size that cant be resized.
    cv2.namedWindow('Webcam Feed', cv2.WINDOW_AUTOSIZE)
    cv2.resizeWindow('Webcam Feed', width, height)

    l_a_muscle = None
    r_a_muscle = None

    # Loop through the frames
    while cap.isOpened():  # and not stop_button_pressed:
        # Read a frame
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, conf=0.75, verbose=False)[0]

            # Visualize the results on the frame
            annotated_frame = results.plot()

            # Extract keypoints
            try:
                # left 6 8 och 10
                # right 7 9 och 11
                # Get coordinates
                l_shoulder = [results.keypoints.data[0][5][0].item(),
                              results.keypoints.data[0][5][1].item()]
                l_elbow = [results.keypoints.data[0][7][0].item(),
                           results.keypoints.data[0][7][1].item()]
                l_wrist = [results.keypoints.data[0][9][0].item(),
                           results.keypoints.data[0][9][1].item()]

                r_shoulder = [results.keypoints.data[0][6][0].item(),
                              results.keypoints.data[0][6][1].item()]
                r_elbow = [results.keypoints.data[0][8][0].item(),
                           results.keypoints.data[0][8][1].item()]
                r_wrist = [results.keypoints.data[0][10][0].item(),
                           results.keypoints.data[0][10][1].item()]

                # Calculate angle
                l_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
                r_angle = calculate_angle(r_shoulder, r_elbow, r_wrist)

                # Visualize angle
                '''
                cv2.putText(annotated_frame, str(l_angle),
                            tuple(np.array(l_elbow, dtype=int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA
                            )
                '''
                # Left arm logic
                if l_angle > 160:
                    l_a_muscle = 'Left Triceps'
                if l_angle < 30 and l_a_muscle == 'Left Triceps':
                    l_a_muscle = 'Left Biceps'

                # Right arm logic
                if r_angle > 160:
                    r_a_muscle = 'Right Triceps'
                if r_angle < 30 and r_a_muscle == 'Right Triceps':
                    r_a_muscle = 'Right Biceps'

            except:
                pass

            # Background box
            cv2.rectangle(annotated_frame, (0, 0), (240, 60), (255, 128, 128), -1)
            cv2.rectangle(annotated_frame, (400, 0), (640, 60), (255, 128, 128), -1)

            # Stage data
            cv2.putText(annotated_frame, 'Active muscle', (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(annotated_frame, 'Active muscle', (405, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.putText(annotated_frame, r_a_muscle, (5, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            cv2.putText(annotated_frame, l_a_muscle, (405, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            # Display the frame with the annotated frame
            cv2.imshow('Webcam Feed', annotated_frame)

            # Check for the 'Esc' key or window close button to break the loop
            if cv2.waitKey(1) == 27 or cv2.getWindowProperty('Webcam Feed', cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            # Break the loop if not successfully reading any frame.
            break

    # Release the captured object and destroy all windows
    cap.release()
    cv2.destroyAllWindows()


# Call the function to open the webcam
open_webcam()
