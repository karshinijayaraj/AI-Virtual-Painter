import cv2
import mediapipe as mp
import numpy as np
from datetime import datetime

# MediaPipe Setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# Camera
cap = cv2.VideoCapture(0)

canvas = None
timestamp = 0

prev_x = 0
prev_y = 0

draw_color = (0, 0, 255)  # Red default

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    # Top Color Menu
    cv2.rectangle(frame, (0, 0), (100, 60), (0, 0, 255), -1)       # Red
    cv2.rectangle(frame, (100, 0), (200, 60), (0, 255, 0), -1)     # Green
    cv2.rectangle(frame, (200, 0), (300, 60), (255, 0, 0), -1)     # Blue
    cv2.rectangle(frame, (300, 0), (400, 60), (0, 255, 255), -1)   # Yellow
    cv2.rectangle(frame, (400, 0), (550, 60), (50, 50, 50), -1)    # Clear

    cv2.putText(frame, "CLEAR", (425, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    timestamp += 1

    if result.hand_landmarks:

        hand = result.hand_landmarks[0]

        index_tip = hand[8]
        index_pip = hand[6]

       smooth_x = previous_x + (current_x - previous_x) / 5
       smooth_y = previous_y + (current_y - previous_y) / 5

        cv2.circle(frame, (x, y), 10, draw_color, -1)

        # Finger Up = Drawing Mode
        if index_tip.y < index_pip.y:

            # Color Selection Area
            if y < 60:

                if 0 < x < 100:
                    draw_color = (0, 0, 255)

                elif 100 < x < 200:
                    draw_color = (0, 255, 0)

                elif 200 < x < 300:
                    draw_color = (255, 0, 0)

                elif 300 < x < 400:
                    draw_color = (0, 255, 255)

                elif 400 < x < 550:
                    canvas = np.zeros_like(frame)

                prev_x = 0
                prev_y = 0

            else:

                if prev_x == 0 and prev_y == 0:
                    prev_x = x
                    prev_y = y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    draw_color,
                    5
                )

                prev_x = x
                prev_y = y

        else:
            prev_x = 0
            prev_y = 0

    output = cv2.add(frame, canvas)

    cv2.putText(
        output,
        "S = Save | Q = Quit",
        (10, output.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("AI Virtual Painter", output)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):

        filename = f"drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        cv2.imwrite(
            filename,
            output
        )

        print(f"Saved: {filename}")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()