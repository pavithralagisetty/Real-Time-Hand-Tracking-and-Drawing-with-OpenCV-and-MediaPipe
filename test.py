import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
#print(frame)

canvas = np.zeros_like(frame)
px, py = -1, -1

color = (0, 0, 255)  # Initial color is red

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
) as hands:

    while True:
        ret, frame = cap.read()
        print(ret)

        if ret:
            frame = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        if id == 8:  # Index finger landmark
                            cv2.circle(frame, (cx, cy), 5, color, -1)

                            if canvas is not None and px != -1 and py != -1:
                                cv2.line(canvas, (px, py), (cx, cy), color, 2)

                            px, py = cx, cy

            cv2.imshow("Frame", frame)
            cv2.imshow("Canvas", canvas)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('c'):  # Press 'c' key to toggle between red and blue colors
                color = (255, 0, 0) if color == (0, 0, 255) else (0, 0, 255)

cap.release()
cv2.destroyAllWindows()
