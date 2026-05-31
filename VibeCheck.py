import argparse
import os
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Constants

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

FINGERTIPS = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

FACE_KEY_POINTS = {
    33: "L_eye_outer", 133: "L_eye_inner",
    362: "R_eye_outer", 263: "R_eye_inner",
    159: "L_eye_top",   145: "L_eye_bot",
    386: "R_eye_top",   374: "R_eye_bot",
    70:  "L_brow",      300: "R_brow",
    1:   "nose_tip",    4:   "nose_base",
    61:  "L_lip",       291: "R_lip",
    0:   "lip_top",     17:  "lip_bot",
    10:  "forehead",    152: "chin",
    234: "L_cheek",     454: "R_cheek",
}

FACE_FEATURE_CONNECTIONS = {
    "left_eye":  [33, 160, 158, 133, 153, 144, 33],
    "right_eye": [362, 385, 387, 263, 373, 380, 362],
    "left_brow": [46, 53, 52, 65, 55],
    "right_brow":[285, 295, 282, 283, 276],
    "outer_lips":[61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                  375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    "inner_lips":[78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
                  324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
    "nose":      [168, 6, 197, 195, 5, 4, 1, 19, 94],
}

FEATURE_COLORS = {
    "left_eye":   (0, 255, 255),
    "right_eye":  (0, 255, 255),
    "left_brow":  (255, 200, 0),
    "right_brow": (255, 200, 0),
    "outer_lips": (0, 80, 255),
    "inner_lips": (0, 140, 255),
    "nose":       (180, 255, 100),
}

GESTURE_IMAGES = {
    "Absolute Cinema": "images/AbsoluteCinema.png",
    "Erm":             "images/ErmDog.jpg",
    "Hang Twenty":     "images/HangTwenty.png",
    "Infinite Void":   "images/InfiniteVoid.png",
    "MonkeyThink":     "images/ThinkingMonkey.jpeg",
    "NaNaNaNaNaNa":    "images/NaNaNaNaNaNa.jpg",
    "Drinking":        "images/beerguy.jpg",
}


# Image helpers

def load_gesture_images():
    cache = {}
    for gesture, path in GESTURE_IMAGES.items():
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            cache[gesture] = img
        else:
            print(f"[WARN] Could not load gesture image: {path}")
    return cache


def draw_gesture_preview(frame, gesture_name, image_cache, size=180):
    if gesture_name not in image_cache:
        return
    fh, fw = frame.shape[:2]
    thumb   = cv2.resize(image_cache[gesture_name], (size, size))
    margin  = 12
    x_off   = fw - size - margin
    y_off   = fh - size - margin
    roi     = frame[y_off:y_off + size, x_off:x_off + size]

    if thumb.shape[2] == 4:
        alpha   = thumb[:, :, 3:4] / 255.0
        blended = (thumb[:, :, :3] * alpha + roi * (1 - alpha)).astype("uint8")
    else:
        blended = thumb

    frame[y_off:y_off + size, x_off:x_off + size] = blended
    cv2.rectangle(frame, (x_off - 1, y_off - 1),
                  (x_off + size, y_off + size), (255, 255, 255), 1)
    cv2.putText(frame, gesture_name, (x_off, y_off - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


# This currently doesnt work for palm detection so it currently works if palm
# is facing camera and when it the other way around

def palm_normal(landmarks):
    """
    Compute the palm surface normal using two edge vectors of the palm quad:
      wrist(0) → index_mcp(5)  and  wrist(0) → pinky_mcp(17)
    The cross product gives a vector perpendicular to the palm.
    A positive z-component means the normal points toward the camera
    (palm facing camera); negative means the back of the hand faces camera.
    MediaPipe x/y are normalised [0,1]; z is in the same scale so the
    cross-product sign is reliable even though the magnitude is not.
    """
    def v(lm): return np.array([lm.x, lm.y, lm.z])

    wrist     = v(landmarks[0])
    idx_mcp   = v(landmarks[5])
    pinky_mcp = v(landmarks[17])

    vec_a = idx_mcp   - wrist   # wrist → index knuckle
    vec_b = pinky_mcp - wrist   # wrist → pinky knuckle

    normal = np.cross(vec_a, vec_b)
    return normal


def is_palm_facing_camera(landmarks):
    """
    Returns True when the palm faces the camera.
    The normal's z-component sign tells us: in MediaPipe's coordinate
    system (y points DOWN), a palm-forward pose gives a normal with
    positive z.  After the image is flipped horizontally (cv2.flip)
    left/right swap but z remains the same convention.
    """
    normal = palm_normal(landmarks)
    return normal[2] > 0          # positive z → palm toward camera


def is_palm_facing_away(landmarks):
    normal = palm_normal(landmarks)
    return normal[2] < 0          # negative z → back of hand toward camera


# Finger/Hand state

def finger_is_up(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y


def landmark_xy(landmarks, index):
    lm = landmarks[index]
    return np.array([lm.x, lm.y])


def point_to_segment_distance(point, start, end):
    segment = end - start
    segment_len_sq = np.dot(segment, segment)
    if segment_len_sq == 0:
        return np.linalg.norm(point - start)

    t = np.dot(point - start, segment) / segment_len_sq
    t = np.clip(t, 0, 1)
    closest = start + t * segment
    return np.linalg.norm(point - closest)


def is_palm_open(landmarks):
    return all([
        finger_is_up(landmarks, 8,  6),
        finger_is_up(landmarks, 12, 10),
        finger_is_up(landmarks, 16, 14),
        finger_is_up(landmarks, 20, 18),
    ])


def thumb_is_extended(landmarks, handedness):
    """Thumb spread sideways (not curled into the palm)."""
    tip  = landmarks[4]
    base = landmarks[2]
    if handedness == "Right":
        return tip.x < base.x
    else:
        return tip.x > base.x


def finger_is_extended(landmarks, mcp, pip, tip):
    wrist = landmark_xy(landmarks, 0)
    mcp_pt = landmark_xy(landmarks, mcp)
    pip_pt = landmark_xy(landmarks, pip)
    tip_pt = landmark_xy(landmarks, tip)

    return (
        np.linalg.norm(tip_pt - wrist) > np.linalg.norm(pip_pt - wrist) * 1.08
        and np.linalg.norm(tip_pt - mcp_pt) > np.linalg.norm(pip_pt - mcp_pt) * 1.10
    )


def is_hang_ten(landmarks, handedness):
    return (
        thumb_is_extended(landmarks, handedness)
        and finger_is_extended(landmarks, 17, 18, 20)
        and not finger_is_extended(landmarks, 5, 6, 8)
        and not finger_is_extended(landmarks, 9, 10, 12)
        and not finger_is_extended(landmarks, 13, 14, 16)
    )


def index_finger_is_straight(landmarks):
    index_mcp = landmark_xy(landmarks, 5)
    index_tip = landmark_xy(landmarks, 8)
    index_pip = landmark_xy(landmarks, 6)
    index_dip = landmark_xy(landmarks, 7)
    index_len = max(np.linalg.norm(index_tip - index_mcp), 0.001)

    pip_offset = point_to_segment_distance(index_pip, index_mcp, index_tip)
    dip_offset = point_to_segment_distance(index_dip, index_mcp, index_tip)

    return (
        finger_is_up(landmarks, 8, 6)
        and pip_offset < index_len * 0.18
        and dip_offset < index_len * 0.18
    )


def is_infinite_void_pose(landmarks):
    index_mcp = landmark_xy(landmarks, 5)
    index_tip = landmark_xy(landmarks, 8)
    index_pip = landmark_xy(landmarks, 6)
    middle_mcp = landmark_xy(landmarks, 9)
    middle_pip = landmark_xy(landmarks, 10)
    middle_dip = landmark_xy(landmarks, 11)
    middle_tip = landmark_xy(landmarks, 12)
    palm_width = max(
        np.linalg.norm(landmark_xy(landmarks, 5) - landmark_xy(landmarks, 17)),
        0.001,
    )
    index_len = max(np.linalg.norm(index_tip - index_mcp), 0.001)
    middle_len = max(np.linalg.norm(middle_tip - middle_mcp), 0.001)

    middle_close_to_index = (
        np.linalg.norm(middle_tip - index_tip) < palm_width * 0.28
        or min(
            point_to_segment_distance(middle_dip, index_pip, index_tip),
            point_to_segment_distance(middle_tip, index_pip, index_tip),
        ) < palm_width * 0.35
    )
    middle_bent = (
        point_to_segment_distance(middle_pip, middle_mcp, middle_tip) > middle_len * 0.12
        or point_to_segment_distance(middle_dip, middle_mcp, middle_tip) > middle_len * 0.12
    )
    middle_less_extended = middle_len < index_len * 0.92
    middle_depth_offset = abs(landmarks[12].z - landmarks[8].z) > palm_width * 0.10
    middle_crossing_index = (
        middle_close_to_index
        and (middle_bent or middle_less_extended or middle_depth_offset)
    )

    return (
        index_finger_is_straight(landmarks)
        and middle_crossing_index
        and not finger_is_up(landmarks, 16, 14)
        and not finger_is_up(landmarks, 20, 18)
    )


# Face geometry

def get_face_bbox(face_landmarks, w, h, padding=0.05):
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    return (
        int(min(xs) * w - padding * w),
        int(min(ys) * h - padding * h),
        int(max(xs) * w + padding * w),
        int(max(ys) * h + padding * h),
    )


def get_lip_bbox(face_landmarks, w, h, padding=0.03):
    """
    Tight bounding box around the lip region using outer-lip landmark indices.
    """
    lip_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
                   375, 321, 405, 314, 17, 84, 181, 91, 146]
    xs = [face_landmarks[i].x for i in lip_indices]
    ys = [face_landmarks[i].y for i in lip_indices]
    return (
        int(min(xs) * w - padding * w),
        int(min(ys) * h - padding * h),
        int(max(xs) * w + padding * w),
        int(max(ys) * h + padding * h),
    )


def get_ear_region(face_landmarks, w, h, side="left"):
    """
    Approximate ear/temple region for NaNaNaNaNaNa detection.
    'left'  uses left cheek / temple landmarks (234, 127, 162, 21)
    'right' uses right cheek / temple landmarks (454, 356, 389, 251)
    Returns (x_min, y_min, x_max, y_max) with generous padding.
    """
    if side == "left":
        indices = [234, 127, 162, 21]
    else:
        indices = [454, 356, 389, 251]

    xs = [face_landmarks[i].x for i in indices]
    ys = [face_landmarks[i].y for i in indices]
    pad = 0.10
    return (
        int((min(xs) - pad) * w),
        int((min(ys) - pad) * h),
        int((max(xs) + pad) * w),
        int((max(ys) + pad) * h),
    )


def landmark_in_bbox(lm, w, h, bbox):
    px, py = int(lm.x * w), int(lm.y * h)
    x0, y0, x1, y1 = bbox
    return x0 <= px <= x1 and y0 <= py <= y1


def index_tip_in_bbox(hand_landmarks, w, h, bbox):
    return landmark_in_bbox(hand_landmarks[8], w, h, bbox)


def thumb_tip_in_bbox(hand_landmarks, w, h, bbox):
    return landmark_in_bbox(hand_landmarks[4], w, h, bbox)


# Per-hand gesture 

def detect_gesture(landmarks, handedness="Right"):
    thumb_tip = landmarks[4]
    thumb_ip  = landmarks[3]
    thumb_up  = (thumb_tip.x > thumb_ip.x if handedness == "Right"
                 else thumb_tip.x < thumb_ip.x)

    index_up  = finger_is_up(landmarks, 8,  6)
    middle_up = finger_is_up(landmarks, 12, 10)
    ring_up   = finger_is_up(landmarks, 16, 14)
    pinky_up  = finger_is_up(landmarks, 20, 18)
    thumb_ext = thumb_is_extended(landmarks, handedness)

    fingers = [thumb_up, index_up, middle_up, ring_up, pinky_up]
    count   = fingers.count(True)

    # Drinking: pinky up, thumb extended, other fingers curled
    # (thumb near mouth checked in caller with face data)
    if pinky_up and not index_up and not middle_up and not ring_up:
        return "Drinking_candidate"

    # Infinite Void: index straight with the middle finger crossing near it.
    if is_infinite_void_pose(landmarks):
        return "Infinite Void"

    # Erm: index up + thumb visibly extended, others curled
    if index_up and thumb_ext and not middle_up and not ring_up and not pinky_up:
        return "Erm"

    if index_up and not any([middle_up, ring_up, pinky_up]):             return "Pointing"

    return f"{count} fingers"


# Two-hand gestures

def detect_two_hand_gesture(hand_landmarks_list, handedness_list, face_landmarks_list, w, h):
    if len(hand_landmarks_list) < 2:
        return None

    lms_a, lms_b = hand_landmarks_list[0], hand_landmarks_list[1]
    hand_a = handedness_list[0] if len(handedness_list) > 0 else "Right"
    hand_b = handedness_list[1] if len(handedness_list) > 1 else "Left"

    thumb_near_head = False

    if is_hang_ten(lms_a, hand_a) and is_hang_ten(lms_b, hand_b):
        return "Hang Twenty"

    if is_palm_open(lms_a) and is_palm_open(lms_b):
        if face_landmarks_list:
            face = face_landmarks_list[0]
            left_ear  = get_ear_region(face, w, h, "left")
            right_ear = get_ear_region(face, w, h, "right")

            hand_a_near = (thumb_tip_in_bbox(lms_a, w, h, left_ear) or
                           thumb_tip_in_bbox(lms_a, w, h, right_ear))
            hand_b_near = (thumb_tip_in_bbox(lms_b, w, h, left_ear) or
                           thumb_tip_in_bbox(lms_b, w, h, right_ear))

            thumb_near_head = hand_a_near and hand_b_near

        if thumb_near_head:
            return "NaNaNaNaNaNa"
        else:
            return "Absolute Cinema"

    return None


# Face drawing

def draw_face_landmarks(frame, face_landmarks, w, h):
    pts = {i: (int(lm.x * w), int(lm.y * h)) for i, lm in enumerate(face_landmarks)}

    for feature, indices in FACE_FEATURE_CONNECTIONS.items():
        color = FEATURE_COLORS[feature]
        poly  = [pts[i] for i in indices if i in pts]
        for j in range(len(poly) - 1):
            cv2.line(frame, poly[j], poly[j + 1], color, 1)

    for idx, label in FACE_KEY_POINTS.items():
        if idx not in pts:
            continue
        x, y = pts[idx]
        cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        cv2.putText(frame, label, (x + 4, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

    return pts


def estimate_head_pose(pts, w, h):
    if not all(k in pts for k in [1, 234, 454]):
        return None
    nose_x        = pts[1][0]
    l_cheek_x     = pts[234][0]
    r_cheek_x     = pts[454][0]
    face_center_x = (l_cheek_x + r_cheek_x) / 2
    offset        = nose_x - face_center_x
    threshold     = (r_cheek_x - l_cheek_x) * 0.1

    if offset < -threshold: return "Looking Left"
    if offset >  threshold: return "Looking Right"
    return "Looking Center"


# HUD 

def draw_hud_hint(frame, show_skeleton, use_face_recognition=False):
    fh, fw   = frame.shape[:2]
    font     = cv2.FONT_HERSHEY_SIMPLEX
    state    = "ON" if show_skeleton else "OFF"
    sk_color = (0, 255, 100) if show_skeleton else (0, 80, 255)
    sk_text  = f"[S] Skeleton: {state}"
    (tw, th), _ = cv2.getTextSize(sk_text, font, 0.5, 1)
    cv2.putText(frame, sk_text, (fw - tw - 10, th + 8), font, 0.5, sk_color, 1)
    if use_face_recognition:
        hint = "[E] Enroll face"
        (ew, _), _ = cv2.getTextSize(hint, font, 0.5, 1)
        cv2.putText(frame, hint, (fw - ew - 10, th + 28), font, 0.5, (180, 180, 255), 1)


def draw_banner(frame, text, color=(0, 215, 255)):
    fh, fw = frame.shape[:2]
    (bw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.4, 3)
    bx = (fw - bw) // 2
    by = 70
    cv2.putText(frame, text, (bx + 2, by + 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 4)
    cv2.putText(frame, text, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)


# Known-face enrollment

def next_enrollment_path(directory="known_faces"):
    os.makedirs(directory, exist_ok=True)
    existing = [
        f for f in os.listdir(directory)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    indices = []
    for f in existing:
        stem = os.path.splitext(f)[0]
        if stem.isdigit():
            indices.append(int(stem))
    next_idx = max(indices) + 1 if indices else 0
    return os.path.join(directory, f"{next_idx:03d}.jpg")


def enroll_face(frame, face_lms_list, fw, fh):
    if not face_lms_list:
        print("[INFO] No face detected — enrollment skipped.")
        return
    bbox = get_face_bbox(face_lms_list[0], fw, fh, padding=0.08)
    x0, y0, x1, y1 = bbox
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(fw, x1), min(fh, y1)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        print("[INFO] Face crop was empty — enrollment skipped.")
        return
    path = next_enrollment_path()
    cv2.imwrite(path, crop)
    print(f"[INFO] Face saved to {path}")


# Main

def main():
    parser = argparse.ArgumentParser(description="VibeCheck")
    parser.add_argument("--face-recognition", action="store_true",
                        help="Enable facial recognition (default: off)")
    args = parser.parse_args()

    use_face_recognition = args.face_recognition

    if use_face_recognition:
        from face_recognition_module import FaceRecognizer, draw_recognition_result
        recognizer = FaceRecognizer()
    else:
        recognizer = None

    gesture_images = load_gesture_images()
    show_skeleton  = True

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="hand_landmarker.task"),
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    face_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="face_landmarker.task"),
        output_face_blendshapes=True,
        num_faces=2,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with (vision.HandLandmarker.create_from_options(hand_options) as hand_landmarker,
          vision.FaceLandmarker.create_from_options(face_options) as face_landmarker):

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame    = cv2.flip(frame, 1)
            fh, fw   = frame.shape[:2]
            rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_landmarker.detect(mp_image)
            face_result = face_landmarker.detect(mp_image)

            face_lms_list = face_result.face_landmarks if face_result.face_landmarks else []
            lip_bboxes    = [get_lip_bbox(fl, fw, fh) for fl in face_lms_list]

            if use_face_recognition and show_skeleton:
                for (top, right, bottom, left), name, matched in recognizer.update(rgb):
                    draw_recognition_result(frame, top, right, bottom, left, name, matched)

            active_two_hand_gesture = None
            per_hand_gestures       = []
            num_hands               = len(hand_result.hand_landmarks) if hand_result.hand_landmarks else 0

            if hand_result.hand_landmarks:
                all_lms = hand_result.hand_landmarks
                all_h   = hand_result.handedness or []

                active_two_hand_gesture = detect_two_hand_gesture(
                    all_lms,
                    [hh[0].category_name for hh in all_h] if all_h else [],
                    face_lms_list,
                    fw, fh,
                )

                for hand_index, hand_landmarks in enumerate(all_lms):
                    points = [(int(lm.x * fw), int(lm.y * fh)) for lm in hand_landmarks]

                    handedness = "Right"
                    if all_h and len(all_h) > hand_index:
                        handedness = all_h[hand_index][0].category_name

                    if show_skeleton:
                        for start, end in HAND_CONNECTIONS:
                            cv2.line(frame, points[start], points[end], (255, 0, 0), 2)
                        for px, py in points:
                            cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
                        for idx, name in FINGERTIPS.items():
                            px, py = points[idx]
                            cv2.putText(frame, name, (px, py - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                            cv2.circle(frame, (px, py), 7, (0, 255, 0), -1)

                    if active_two_hand_gesture:
                        gesture = "Open Hand"
                    else:
                        gesture = detect_gesture(hand_landmarks, handedness)

                        if num_hands == 1:
                            if gesture == "Drinking_candidate":
                                if lip_bboxes and thumb_tip_in_bbox(hand_landmarks, fw, fh, lip_bboxes[0]):
                                    gesture = "Drinking"
                                else:
                                    gesture = "Call Me"

                            elif gesture == "Erm":
                                if lip_bboxes and index_tip_in_bbox(hand_landmarks, fw, fh, lip_bboxes[0]):
                                    gesture = "MonkeyThink"

                            elif gesture == "Pointing":
                                if lip_bboxes and index_tip_in_bbox(hand_landmarks, fw, fh, lip_bboxes[0]):
                                    gesture = "MonkeyThink"

                        else:
                            if gesture in ("Erm", "MonkeyThink", "Drinking_candidate", "Drinking"):
                                gesture = detect_gesture.__wrapped__(hand_landmarks, handedness) \
                                    if hasattr(detect_gesture, "__wrapped__") else "—"

                    per_hand_gestures.append((gesture, points[0], handedness))

                    if show_skeleton:
                        wrist_x, wrist_y = points[0]
                        cv2.putText(frame, f"{handedness}: {gesture}",
                                    (wrist_x - 40, wrist_y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            BANNER_GESTURES = {"Erm", "Hang Twenty", "Infinite Void", "MonkeyThink", "Drinking",
                               "Absolute Cinema", "NaNaNaNaNaNa"}

            special = active_two_hand_gesture
            if not special:
                for g, _, _ in per_hand_gestures:
                    if g in BANNER_GESTURES:
                        special = g
                        break

            if special:
                draw_banner(frame, special.upper())
                draw_gesture_preview(frame, special, gesture_images)

            if face_result.face_landmarks:
                for face_index, face_landmarks in enumerate(face_result.face_landmarks):
                    if show_skeleton:
                        pts  = draw_face_landmarks(frame, face_landmarks, fw, fh)
                        pose = estimate_head_pose(pts, fw, fh)
                        if pose and 10 in pts:
                            cv2.putText(frame, pose,
                                        (pts[10][0] - 40, pts[10][1] - 15),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

                    if (face_result.face_blendshapes
                            and len(face_result.face_blendshapes) > face_index
                            and show_skeleton):
                        blendshapes = face_result.face_blendshapes[face_index]
                        interesting = sorted(
                            [b for b in blendshapes
                             if b.score > 0.4 and b.category_name not in {
                                 "eyeLookDownLeft", "eyeLookDownRight",
                                 "eyeLookInLeft",   "eyeLookInRight",
                                 "eyeLookOutLeft",  "eyeLookOutRight",
                                 "eyeLookUpLeft",   "eyeLookUpRight"}],
                            key=lambda b: b.score, reverse=True
                        )
                        for i, b in enumerate(interesting[:3]):
                            cv2.putText(frame,
                                        f"{b.category_name}: {b.score:.2f}",
                                        (10, 30 + i * 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 255, 180), 1)

            draw_hud_hint(frame, show_skeleton, use_face_recognition)
            cv2.imshow("VibeCheck", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                show_skeleton = not show_skeleton
            elif key == ord("e"):
                enroll_face(frame, face_lms_list, fw, fh)
                if use_face_recognition and recognizer is not None:
                    from face_recognition_module import load_known_faces
                    recognizer.known_encodings, recognizer.known_names = load_known_faces()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()