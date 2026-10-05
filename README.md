# visual-cues

Video visual cue extraction: facial expressions, FACS Action Units / blendshapes, head pose, gaze tracking, body language, and posture analysis.

Part of the **Roosevelt Advisors** intelligence stack, designed to accompany `transcribe-core` for multimodal video recording analysis.

---

## Capabilities

1. **Facial Affect & Expression Analysis**:
   - Discrete expressions: Joy, neutral, sadness, frown, surprise, speaking.
   - Affective valence score: $-1.0$ (negative/frowning) to $+1.0$ (positive/smiling).
   - FACS Action Units / Blendshapes:
     - `AU06_12_smile`: Cheek raiser and lip corner puller intensity ($0.0 - 1.0$).
     - `AU04_brow_furrow`: Brow lowerer (concentration, confusion, skepticism).
     - `AU01_02_brow_raise`: Eyebrow raiser (alertness, astonishment, surprise).
     - `AU25_26_mouth_open`: Mouth open / jaw drop (speech, laughing).
     - `AU07_eye_squint`: Eyelid tightening (critical evaluation).

2. **Body Posture & Movement Dynamics**:
   - Torso inclination / Lean: `forward` (attentive, engaged), `upright` (neutral), `backward` (reclined, skeptical).
   - Posture openness: `open` vs `closed` (defensive, arms crossed).
   - Slouching severity score ($0.0 - 1.0$).
   - Shoulder tension / shrug indicator ($0.0 - 1.0$).
   - Restlessness / Fidgeting index ($0.0 - 1.0$).

3. **Head Pose, Gestures & Gaze**:
   - 3D Head Pose: `pitch` (up/down), `yaw` (left/right), `roll` (tilt).
   - Head Gestures: Nodding (affirmation, active listening), shaking (negation), tilting (curiosity).
   - Gaze Tracking: `direct` (eye contact with camera/screen), `looking_down` (reading notes), `looking_away`.

4. **Turn-Aligned Multimodal Fusion**:
   - When paired with audio turn intervals from `transcribe-core`, aggregates visual cues over each speaker's dialogue turns.

---

## Installation

```bash
pip install .
```

---

## Quick Start (CLI)

```bash
# Analyze video recording with default 2 FPS sampling
visual-cues --video meeting.mp4 --out summary.txt --json cues.json

# Fast sampling with custom FPS
visual-cues --video interview.mp4 --fps 1.0 --json cues.json
```

---

## Python API

```python
from visual_cues import VisualCuesPipeline

pipeline = VisualCuesPipeline()

# Analyze a video file
result = pipeline.analyze_video(
    "recording.mp4",
    sample_fps=2.0,
    turn_intervals=[(0.0, 15.2), (15.2, 32.5)],  # Optional speech turns
)

print("Video Affect Valence:", result.video_summary["avg_valence"])
print("Eye Contact Ratio:", result.video_summary["eye_contact_ratio"])

# Inspect turn-aligned visual cues
for turn in result.turn_summaries:
    print(f"[{turn.start:.1f} - {turn.end:.1f}] {turn.dominant_expression} | {turn.posture_lean} | {turn.posture_openness}")
```

---

## Testing

```bash
pytest tests -v
```
