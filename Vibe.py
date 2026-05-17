import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm
]

FINGERTIPS = {
    4: "thumb",
    8: "index",
    12: "middle",
    16: "ring",
    20: "pinky",
}


def finger_is_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


def detect_gesture(landmarks, handedness="Right"):
    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]

    if handedness == "Right":
        thumb_up = thumb_tip.x > thumb_ip.x
    else:
        thumb_up = thumb_tip.x < thumb_ip.x

    index_up = finger_is_up(landmarks, 8, 6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up = finger_is_up(landmarks, 16, 14)
    pinky_up = finger_is_up(landmarks, 20, 18)

    fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    count = fingers.count(True)

    if count == 0:
        return "Fist"
    if count == 5:
        return "Open Hand"
    if index_up and middle_up and not ring_up and not pinky_up:
        return "Peace"
    if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
        return "Thumbs Up"
    if index_up and not middle_up and not ring_up and not pinky_up:
        return "Pointing"
    if thumb_up and pinky_up and not index_up and not middle_up and not ring_up:
        return "Call Me"

    return f"{count} fingers"

def main():
    base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand_index, hand_landmarks in enumerate(result.hand_landmarks):
                    points = []

                    for lm in hand_landmarks:
                        x, y = int(lm.x * w), int(lm.y * h)
                        points.append((x, y))

                    # Draw skeleton lines
                    for start, end in HAND_CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (255, 0, 0),
                            2
                        )

                    # Draw all joint dots
                    for x, y in points:
                        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

                    # Label fingertips
                    for idx, name in FINGERTIPS.items():
                        x, y = points[idx]
                        cv2.putText(
                            frame,
                            name,
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1
                        )
                        cv2.circle(frame, (x, y), 7, (0, 255, 0), -1)

                    # Get handedness if available
                    handedness = "Right"
                    if result.handedness and len(result.handedness) > hand_index:
                        handedness = result.handedness[hand_index][0].category_name

                    gesture = detect_gesture(hand_landmarks, handedness)

                    wrist_x, wrist_y = points[0]
                    cv2.putText(
                        frame,
                        f"{handedness}: {gesture}",
                        (wrist_x - 40, wrist_y - 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

            cv2.imshow("VibeCheck", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()