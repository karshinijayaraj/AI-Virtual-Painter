import cv2
import mediapipe as mp
import numpy as np
import time

# -----------------------------
# MediaPipe Hand Landmarker
# -----------------------------
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

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

timestamp = 0

# -----------------------------
# Smoothing
# -----------------------------
smooth_x = 0
smooth_y = 0

# -----------------------------
# Drawing
# -----------------------------
canvas = None

prev_draw_x = 0
prev_draw_y = 0

# -----------------------------
# FPS
# -----------------------------
prev_time = 0

# -----------------------------
# Hand Skeleton Connections
# -----------------------------
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

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

        points = []

        # -----------------------------
        # Draw landmarks
        # -----------------------------
        for lm in hand:

            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])

            points.append((x, y))

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # -----------------------------
        # Draw skeleton
        # -----------------------------
        for start, end in connections:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 0, 0),
                2
            )

        # -----------------------------
        # Index finger tip
        # -----------------------------
        current_x, current_y = points[8]

        if smooth_x == 0 and smooth_y == 0:
            smooth_x = current_x
            smooth_y = current_y

        # Smoothing
        smooth_x = int(
            smooth_x + (current_x - smooth_x) / 3
        )

        smooth_y = int(
            smooth_y + (current_y - smooth_y) / 3
        )

        # Cursor
        cv2.circle(
            frame,
            (smooth_x, smooth_y),
            15,
            (0, 0, 255),
            -1
        )

        # -----------------------------
        # Draw only when index finger raised
        # -----------------------------
        index_tip = hand[8]
        index_pip = hand[6]

        if index_tip.y < index_pip.y:

            cv2.putText(
                frame,
                "DRAWING",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if prev_draw_x == 0 and prev_draw_y == 0:
                prev_draw_x = smooth_x
                prev_draw_y = smooth_y

            cv2.line(
                canvas,
                (prev_draw_x, prev_draw_y),
                (smooth_x, smooth_y),
                (255, 255, 255),
                5,
                cv2.LINE_AA
            )

            prev_draw_x = smooth_x
            prev_draw_y = smooth_y

        else:

            cv2.putText(
                frame,
                "PAUSED",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            prev_draw_x = 0
            prev_draw_y = 0

    # -----------------------------
    # FPS
    # -----------------------------
    current_time = time.time()

    fps = (
        1 / (current_time - prev_time)
        if prev_time != 0
        else 0
    )

    prev_time = current_time

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "C = Clear | Q = Quit",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # Merge Drawing + Camera
    # -----------------------------
    output = cv2.add(
        frame,
        canvas
    )

    cv2.imshow(
        "Virtual Painter V2",
        output
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()