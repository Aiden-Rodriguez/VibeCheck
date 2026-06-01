import os
import cv2
import numpy as np

try:
    import face_recognition
    _FR_AVAILABLE = True
except ImportError:
    _FR_AVAILABLE = False
    print("[ERROR] face_recognition not installed.")
    print("        Run:  pip install dlib-bin face_recognition")


# Config

KNOWN_FACES_DIR = "known_faces"   # folder of reference images
TOLERANCE       = 0.55            # lower = stricter
CHECK_EVERY_N   = 5               # skip N-1 frames between recognition runs
SCALE           = 0.5             # downscale factor for detection (speed)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

_GREEN = (50, 220, 60)
_RED   = (40, 50, 245)


# Load known faces

def known_face_image_paths(directory: str):
    for fname in sorted(os.listdir(directory)):
        if fname.lower().endswith(IMAGE_EXTENSIONS):
            yield fname, os.path.join(directory, fname)


def display_name_from_filename(filename: str):
    return os.path.splitext(filename)[0].replace("_", " ").title()


def load_known_faces(directory: str = KNOWN_FACES_DIR):
    encodings, names = [], []

    if not _FR_AVAILABLE:
        return encodings, names

    if not os.path.isdir(directory):
        print(f"[WARN] Known-faces directory not found: {directory!r}")
        print(f"       Create it and add reference images:")
        print(f"         known_faces/yourname.jpg")
        return encodings, names

    for fname, path in known_face_image_paths(directory):
        img  = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            name = display_name_from_filename(fname)
            encodings.append(encs[0])
            names.append(name)
            print(f"[INFO] Loaded face: {name}  ({fname})")
        else:
            print(f"[WARN] No face detected in {fname!r} — skipping.")

    if not encodings:
        print("[WARN] No known faces loaded. Add images to known_faces/")

    return encodings, names


class FaceRecognizer:
    """
    Stateful face recogniser with frame-skip caching for real-time performance.

    Parameters
    ----------
    known_dir   : path to folder of reference images
    tolerance   : max face-distance to count as a match (lower = stricter)
    check_every : run full recognition every N frames; return cache otherwise
    scale       : resize factor applied before detection (0.5 = half resolution)
    """

    def __init__(
        self,
        known_dir:   str   = KNOWN_FACES_DIR,
        tolerance:   float = TOLERANCE,
        check_every: int   = CHECK_EVERY_N,
        scale:       float = SCALE,
    ):
        self.known_encodings, self.known_names = load_known_faces(known_dir)
        self.tolerance   = tolerance
        self.check_every = check_every
        self.scale       = scale
        self._cache:     list = []
        self._frame_idx: int  = 0

    def update(self, rgb_frame: np.ndarray) -> list:
        """
        Process one RGB frame.  Returns cached results on skipped frames.

        Returns
        -------
        list of  ((top, right, bottom, left), name, matched)
            box     — face bounding box in *original* frame coordinates
            name    — recognised name, or "Unknown"
            matched — True if a known face matched
        """
        if not _FR_AVAILABLE:
            return []

        self._frame_idx += 1
        if self._frame_idx % self.check_every != 0:
            return self._cache

        # Downscale for speed
        small = cv2.resize(rgb_frame, (0, 0), fx=self.scale, fy=self.scale,
                           interpolation=cv2.INTER_LINEAR)

        locations = face_recognition.face_locations(small, model="hog")
        encodings = face_recognition.face_encodings(small, locations)

        results = []
        for enc, loc in zip(encodings, locations):
            name, matched = self._classify(enc)
            # Scale bounding box back to original resolution
            top, right, bottom, left = [int(v / self.scale) for v in loc]
            results.append(((top, right, bottom, left), name, matched))

        self._cache = results
        return results

    # ------------------------------------------------------------------
    def _classify(self, encoding) -> tuple[str, bool]:
        """Return (name, matched) for a single face encoding."""
        if not self.known_encodings:
            return "Unknown", False

        distances = face_recognition.face_distance(self.known_encodings, encoding)
        best_idx  = int(np.argmin(distances))

        if distances[best_idx] <= self.tolerance:
            return self.known_names[best_idx], True
        return "Unknown", False


# Drawing 

def draw_recognition_result(
    frame: np.ndarray,
    top: int, right: int, bottom: int, left: int,
    name: str,
    matched: bool,
):
    """
    Draw a coloured bounding box, name label, and ✓/✗ symbol on `frame` (BGR).

    Green box + name  →  recognised person
    Red   box + ✗     →  unknown face
    """
    color = _GREEN if matched else _RED

    # Bounding box 
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Name label (below the box)
    label = name
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness  = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_y0 = bottom
    label_y1 = bottom + th + baseline + 6

    cv2.rectangle(frame, (left, label_y0), (left + tw + 10, label_y1), color, -1)
    cv2.putText(frame, label, (left + 5, label_y1 - baseline - 2),
                font, font_scale, (255, 255, 255), thickness)

    # ── ✓ / ✗ symbol centred inside the bounding box ─────────────────
    symbol    = "OK" if matched else "X"    # OpenCV can't render Unicode ✓/✗
    box_w     = right - left
    box_h     = bottom - top
    sym_scale = max(0.7, min(2.5, min(box_w, box_h) / 100))
    (sw, sh), _ = cv2.getTextSize(symbol, font, sym_scale, 3)
    sx = left + (box_w - sw) // 2
    sy = top  + (box_h + sh) // 2

    # Drop shadow for readability
    cv2.putText(frame, symbol, (sx + 2, sy + 2), font, sym_scale, (0, 0, 0),    4)
    cv2.putText(frame, symbol, (sx,     sy),     font, sym_scale, color,         3)


# ── Quick standalone test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if not _FR_AVAILABLE:
        sys.exit(1)

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    recognizer = FaceRecognizer()
    print("[INFO] Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        for (top, right, bottom, left), name, matched in recognizer.update(rgb):
            draw_recognition_result(frame, top, right, bottom, left, name, matched)

        cv2.imshow("Face Recognition Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
