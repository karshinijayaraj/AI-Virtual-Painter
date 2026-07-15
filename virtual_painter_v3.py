import cv2
import mediapipe as mp
import numpy as np
import time
from datetime import datetime


# -----------------------------
# MediaPipe Setup
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
# Drawing Variables
# -----------------------------
canvas = None

prev_x = 0
prev_y = 0

smooth_x = 0
smooth_y = 0

brush_size = 5

# Default color = red
color = (0,0,255)


# -----------------------------
# FPS
# -----------------------------
prev_time = 0


# -----------------------------
# Skeleton Connections
# -----------------------------
connections = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


# -----------------------------
# Colors
# -----------------------------
RED = (0,0,255)
GREEN = (0,255,0)
BLUE = (255,0,0)
YELLOW = (0,255,255)
ERASER = (0,0,0)


# -----------------------------
# Main Loop
# -----------------------------
while True:


    success, frame = cap.read()

    if not success:
        break


    frame = cv2.flip(frame,1)


    if canvas is None:
        canvas = np.zeros_like(frame)



    # -----------------------------
    # Color Palette
    # -----------------------------

    cv2.rectangle(frame,(0,0),(100,70),RED,-1)
    cv2.rectangle(frame,(100,0),(200,70),GREEN,-1)
    cv2.rectangle(frame,(200,0),(300,70),BLUE,-1)
    cv2.rectangle(frame,(300,0),(400,70),YELLOW,-1)
    cv2.rectangle(frame,(400,0),(520,70),(50,50,50),-1)


    cv2.putText(
        frame,
        "ERASE",
        (410,45),
        cv2.FONT_HERSHEY_SIMPLEX,
        .7,
        (255,255,255),
        2
    )


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

        points=[]


        # -----------------------------
        # Skeleton
        # -----------------------------

        for lm in hand:

            x=int(lm.x*frame.shape[1])
            y=int(lm.y*frame.shape[0])

            points.append((x,y))


            cv2.circle(
                frame,
                (x,y),
                5,
                (0,255,0),
                -1
            )


        for a,b in connections:

            cv2.line(
                frame,
                points[a],
                points[b],
                (255,0,0),
                2
            )



        # -----------------------------
        # Smooth Cursor
        # -----------------------------

        x,y = points[8]


        if smooth_x==0:
            smooth_x=x
            smooth_y=y


        smooth_x=int(
            smooth_x+(x-smooth_x)/3
        )

        smooth_y=int(
            smooth_y+(y-smooth_y)/3
        )


        cv2.circle(
            frame,
            (smooth_x,smooth_y),
            12,
            color,
            -1
        )



        # -----------------------------
        # Color Selection
        # -----------------------------

        if smooth_y < 70:


            if smooth_x < 100:
                color=RED

            elif smooth_x < 200:
                color=GREEN

            elif smooth_x < 300:
                color=BLUE

            elif smooth_x < 400:
                color=YELLOW

            elif smooth_x < 520:
                color=ERASER


            prev_x=0
            prev_y=0



        else:


            # Index finger raised = draw

            if hand[8].y < hand[6].y:


                if prev_x==0:
                    prev_x=smooth_x
                    prev_y=smooth_y


                cv2.line(
                    canvas,
                    (prev_x,prev_y),
                    (smooth_x,smooth_y),
                    color,
                    brush_size,
                    cv2.LINE_AA
                )


                prev_x=smooth_x
                prev_y=smooth_y


            else:

                prev_x=0
                prev_y=0



    # -----------------------------
    # FPS
    # -----------------------------

    current=time.time()

    fps=1/(current-prev_time) if prev_time else 0

    prev_time=current


    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20,110),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )


    cv2.putText(
        frame,
        f"Brush: {brush_size}",
        (20,150),
        cv2.FONT_HERSHEY_SIMPLEX,
        .8,
        (255,255,255),
        2
    )


    output=cv2.add(
        frame,
        canvas
    )


    cv2.imshow(
        "AI Virtual Painter V3",
        output
    )


    key=cv2.waitKey(1)&0xff


    if key==ord('c'):
        canvas[:]=0


    if key==ord('s'):

        name=f"drawing_{datetime.now().strftime('%H%M%S')}.png"

        cv2.imwrite(
            name,
            output
        )

        print("Saved:",name)


    if key==ord('+'):
        brush_size+=2


    if key==ord('-') and brush_size>2:
        brush_size-=2


    if key==ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
