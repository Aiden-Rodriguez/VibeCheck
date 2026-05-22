# VibeCheck

VibeCheck is a Python webcam app that uses OpenCV and MediaPipe to track hands
and faces in real time. It recognizes a handful of hand/face-adjacent gestures,
draws landmark overlays, and displays matching reaction images for special
gestures.

The main program is `VibeCheck.py`. Optional face recognition support lives in
`face_recognition_module.py`.

## What It Does

- Opens your webcam and mirrors the video feed.
- Detects up to two hands using MediaPipe's hand landmarker model.
- Detects up to two faces using MediaPipe's face landmarker model.
- Draws hand skeletons, fingertips, face feature outlines, and head pose hints.
- Recognizes popular 'meme' gestures like:
  - `Erm`
  - `Infinite Void`
  - `MonkeyThink`
  - `Drinking`
  - `Absolute Cinema`
  - `NaNaNaNaNaNa`
- Shows a large banner and preview image when a special gesture is detected.
- Optionally recognizes known faces from images stored in `known_faces/`.

## Project Structure

```text
.
├── VibeCheck.py                # Main webcam app
├── face_recognition_module.py  # Optional known-face recognition helper
├── hand_landmarker.task        # MediaPipe hand model
├── face_landmarker.task        # MediaPipe face model
├── images/                     # Gesture preview images
├── known_faces/                # Reference face images for recognition
├── test.py                     # Small scratch/test file
└── vibecheck_env/              # Existing local virtual environment, if present
```

## Setup

Python 3.11 or 3.12 is recommended because computer-vision packages can lag
behind the newest Python releases.

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the core dependencies:

```bash
python -m pip install opencv-python mediapipe numpy
```

If you want optional face recognition, also install:

```bash
python -m pip install cmake dlib face_recognition
```

On some systems, `dlib` may require extra compiler tools. On Windows, you may
need Visual C++ Build Tools:

https://visualstudio.microsoft.com/visual-cpp-build-tools/

## How To Run

Run the normal hand/face landmark and gesture app:

```bash
python VibeCheck.py
```

Run with optional known-face recognition enabled:

```bash
python VibeCheck.py --face-recognition
```

Face recognition may run slower because it performs face encoding and matching
in addition to the MediaPipe landmark detection.

If you want to use the existing virtual environment included in this project
instead of creating a new one, activate it first:

```bash
source vibecheck_env/bin/activate
python VibeCheck.py
```

## Controls

| Key | Action |
| --- | --- |
| `s` | Toggle skeleton/landmark overlays |
| `e` | Enroll the currently detected face into `known_faces/` |
| `q` | Quit |

The `e` enrollment key saves a cropped face image into `known_faces/`. If face
recognition is enabled, the known-face list is reloaded after enrollment.

## How The App Works

1. `VibeCheck.py` opens the webcam with OpenCV.
2. Each frame is flipped horizontally so it behaves like a mirror.
3. The frame is converted from BGR to RGB for MediaPipe.
4. MediaPipe detects hand landmarks and face landmarks.
5. Hand landmarks are interpreted with simple geometry:
   - A finger is considered "up" when its tip is above its lower joint.
   - The thumb is checked sideways based on handedness.
   - `Infinite Void` looks for a straight index finger with the middle finger
     clustered close to it.
   - Two open hands trigger the `Absolute Cinema` gesture.
   - Two open hands near the head trigger `NaNaNaNaNaNa`.
6. Face landmarks are used for:
   - Face and lip bounding boxes.
   - Approximate ear/head regions.
   - Basic head pose estimation.
   - Gesture context, such as detecting a finger near the mouth.
7. If a special gesture is active, the app draws a banner and displays the
   matching image from `images/`.
8. If `--face-recognition` is enabled, `face_recognition_module.py` checks faces
   against reference images in `known_faces/` every few frames and draws labeled
   boxes.

## Known Faces

To use known-face recognition manually, add clear face images to `known_faces/`.
The filename becomes the display name. For example:

```text
known_faces/jimmy.jpg
known_faces/jane_doe.png
```

Those would display as `Jimmy` and `Jane Doe`.

## Notes

- The `.task` model files must stay in the project root unless the paths in
  `VibeCheck.py` are changed.
- The preview image paths are configured in the `GESTURE_IMAGES` dictionary in
  `VibeCheck.py`.
- Camera permissions may need to be enabled in your operating system before
  OpenCV can access the webcam.
