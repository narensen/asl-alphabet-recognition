# ASL Alphabet Recognition

Real-time American Sign Language fingerspelling recognition in Python. Sign letters at your
webcam and they are typed on screen, all 26 letters including the motion letters **J** and **Z**.

- **Hand landmarks, not pixels.** MediaPipe finds 21 3D keypoints on your hand, and a small
  PyTorch network (about 60k parameters) classifies the 24 static letters from them. Background,
  lighting and skin tone matter much less than in pixel-based CNNs, it runs in real time on a
  laptop CPU, and it trains in seconds from about 15 minutes of your own recordings.
- **J and Z from motion.** The fingertip path is tracked and matched against J and Z templates.
- **Hold to type.** Hold a letter for about a second to type it, bounce your hand to repeat a
  letter ("HELLO"), and leave the frame to type a space (optional).
- **Either hand.** Training mirrors samples at random, so left- and right-handed signing both work.

```
webcam frame
   │
   ▼
MediaPipe HandLandmarker ──► 21 landmarks (x, y, z)
   │
   ├─► normalise (wrist at origin, divide by palm size; orientation kept)
   │      └─► 91 features (63 coordinates + 28 finger distances)
   │             └─► MLP 91 → 256 → 128 → 24 ─► smoothing ─► hold-to-type ─► text
   │
   └─► handshape check (pinky only / index only) + fingertip trajectory
          └─► template match ─► J / Z ─────────────────────────────────────► text
```

## Setup

Python 3.9 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

On Linux without an NVIDIA GPU, install the much smaller CPU build of PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

The MediaPipe hand model (about 8 MB) is downloaded to `models/` the first time you run a script.

## Quick start

A starter model is included (see [Results](#results)), so you can try recognition right away:

```bash
python recognize.py
```

For the best accuracy **on your own hand and camera**, record your own data. It takes about
15 minutes:

```bash
python collect.py        # record samples (keys below)
python train.py          # train (usually under a minute on a CPU) -> models/asl_classifier.pt
python recognize.py      # sign away
```

Your recordings go to `data/landmarks.csv` and training replaces the starter model. Copy
`models/asl_classifier.pt` somewhere first if you want to keep it.

### Recording data (`collect.py`)

| Key | Action |
|---|---|
| `A`–`Y` (not `J`) | record one take of that letter (after a short countdown) |
| `0` | record "nothing": relaxed hands and in-between poses (optional, reduces false letters) |
| `SPACE` | cancel the take in progress |
| `BACKSPACE` | cancel the take, or undo the last saved take |
| `ESC` | quit |

Aim for **3–5 takes of 100 samples per letter**, recorded at different moments. Move your hand
a little during each take: nearer, farther, small tilts. `train.py` keeps whole takes together
when it sets aside validation data, so its accuracy reflects recordings the model hasn't seen.

### Recognising (`recognize.py`)

| Key | Action |
|---|---|
| `SPACE` / `BACKSPACE` / `C` | type a space / delete / clear |
| `M` | turn J/Z detection on or off |
| `D` | debug overlay (top-3 guesses, hand speed, handedness) |
| `S` | append the text to `typed_text.txt` |
| `Q` / `ESC` | quit |

Useful options: `--hold 0.6` (type faster), `--min-conf 0.8` (stricter), `--auto-space 1.5`
(space after 1.5 s without a hand), `--camera 1`, `--no-motion`, and
`--video clip.mp4 --save-video out.mp4 --headless` to process a video file.

**Signing tips**
- Hold each letter still; the cyan bar under the letter fills up, then the letter is typed.
- Double letters: after the first one, bounce or slide your hand slightly and hold again.
- **J**: make the I handshape (pinky up) and draw a J with your pinky.
  **Z**: point your index finger and draw a Z. A magenta trail shows the path being tracked.
- Keep your whole hand in the frame, with reasonable light. A plain background helps MediaPipe.

### Training on a public dataset (optional)

`extract_images.py` turns any folder of labelled images (`dataset/A/*.jpg`, `dataset/B/*.jpg`, …)
into a landmark CSV. For example, with the Kaggle
[ASL Alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) dataset:

```bash
python extract_images.py path/to/asl_alphabet_train --out data/kaggle.csv --limit 600 --group-size 100 --workers 4
python train.py --data data/landmarks.csv data/kaggle.csv
```

`--group-size 100` treats runs of 100 consecutive frames as one "take". That dataset was cut
from videos, and a frame-level split would put near-identical frames in both training and
validation. Mixing public data with your own recordings usually gives the most robust model.

To classify single photos: `python predict_image.py photo.jpg --save-dir predictions/`.

## Results

The included starter model was trained on 1,814 hand photos (the ones where MediaPipe found a
hand, out of 2,046) from a public ASL letters dataset with many different signers, cameras and
backgrounds ([atalaydenknalbant/asl-dataset](https://huggingface.co/datasets/atalaydenknalbant/asl-dataset)
on Hugging Face, a Roboflow export). It was evaluated on that dataset's separate validation and
test photos:

| Split | Images | Accuracy | Macro F1 |
|---|---|---|---|
| Validation | 205 | 93.2% | 0.89 |
| Test | 117 | 91.5% | 0.91 |

Most mistakes are between **M, N, S and T**. These fist-shaped letters differ only by where the
thumb sits, and the thumb is often hidden from the camera. `models/confusion_matrix.png` shows
the full picture. The pipeline was also checked end to end on a scripted video built from these
photos (moving them to draw J and Z): it typed `HELLO JAZZ`, including the double L and both
motion letters. Expect the best results after training on your own recordings.

## Project structure

```
collect.py          record landmark samples from the webcam
extract_images.py   build a landmark CSV from a folder of labelled images
train.py            train + evaluate the classifier (writes model, metrics, plots)
recognize.py        real-time recognition app (webcam or video file)
predict_image.py    classify image files
asl/
  landmarks.py      MediaPipe HandLandmarker wrapper, model download, drawing
  features.py       normalisation, 91-d features, augmentation
  model.py          PyTorch MLP, save/load, LetterClassifier
  motion.py         J/Z recogniser (handshape gate + trajectory template matching)
  text.py           probability smoothing, stillness, hold-to-type text builder
  pipeline.py       per-frame logic tying it together (testable without a camera)
  ui.py, video.py   OpenCV drawing, camera and window helpers
tests/              pytest suite (synthetic hands, no camera needed)
```

Run the tests with `pip install pytest` and then `pytest`.

## Design notes

- **Why landmarks?** A CNN trained on a dataset like Sign Language MNIST scores about 99% on its
  own test set but struggles on a live webcam, because the background, lighting and framing all
  differ. Landmarks remove most of that, so a tiny model trained on a little data generalises
  well.
- **What the normalisation keeps.** Position and scale are removed (wrist to origin, divide by
  palm size). Rotation is kept on purpose: H/U, K/P and G/Q differ mainly by orientation.
- **Augmentation.** Small 3D rotations, perspective stretch, landmark jitter and left/right
  mirroring are applied on the fly to the landmarks, not the images.
- **Honest validation.** Splits are made by take, because consecutive frames are near-duplicates.
- **J/Z without training data.** A handshape gate (only the pinky or only the index finger
  extended) plus template matching of the resampled fingertip path. This is the idea behind the
  "$1 unistroke recogniser", with mirrored templates for left-handed signers. A J also has to
  hook (turn at least 45°), so a plain downward swipe of the I handshape isn't read as J.
- **Typing logic.** A letter is typed after being held still for `--hold` seconds. It can't
  repeat until it is released (another letter, no hand, or a bounce of at least 0.25 palm
  widths), and slow drift never counts as a bounce. The I held before drawing a J (or the D
  before a Z) is replaced by the motion letter.

## Troubleshooting

- **Camera doesn't open.** Try `--camera 1`, close other apps that use the camera, and on macOS
  allow camera access for your terminal (System Settings → Privacy & Security → Camera).
- **Keys do nothing.** Click the video window first so it has keyboard focus.
- **Model download fails** (proxy or firewall). Download
  [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
  manually and save it as `models/hand_landmarker.task`.
- **Log lines like `W0000 … inference_feedback_manager` or `landmark_projection_calculator`**
  come from MediaPipe's native code and are harmless.
- **Some letters are unreliable.** Record 2–3 more takes of them (and of the letters they get
  confused with), then retrain. Run `recognize.py --debug` to see the top-3 guesses.
- **J or Z not detected.** Draw the letter bigger (about two hand-widths tall) and in one
  smooth motion. The magenta trail shows what is being tracked. The thresholds are in
  `MotionConfig` in `asl/motion.py`.

## Ideas for extending it

- Replace the J/Z templates with a small GRU or temporal CNN over landmark windows, which could
  also learn whole signs.
- Word suggestions or spell-checking on the typed text, plus text-to-speech.
- Two-handed alphabets (BSL, ISL): concatenate both hands' features (`num_hands=2`).
- Export the MLP to ONNX or TFLite for mobile or web use.
