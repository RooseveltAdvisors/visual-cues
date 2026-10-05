"""Head pose (pitch/yaw/roll), head gestures (nodding/shaking), and gaze tracking."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GazeCues:
    """Head orientation, head gestures, and gaze direction cues."""

    # Head Pose (degrees)
    pitch_deg: float = 0.0  # Up (+), Down (-)
    yaw_deg: float = 0.0  # Left (-), Right (+)
    roll_deg: float = 0.0  # Tilt left (-), Tilt right (+)

    # Head Gestures
    nodding: bool = False  # Affirmative / active listening nod
    shaking: bool = False  # Negation / head shake
    tilted: bool = False  # Head tilt (curiosity, critical thinking)

    # Gaze & Eye Contact
    eye_contact: str = "direct"  # "direct", "looking_down", "looking_away"
    eye_contact_confidence: float = 0.85
    gaze_target: str = "camera / screen"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "head_pose": {
                "pitch_deg": round(self.pitch_deg, 1),
                "yaw_deg": round(self.yaw_deg, 1),
                "roll_deg": round(self.roll_deg, 1),
            },
            "head_gestures": {
                "nodding": self.nodding,
                "shaking": self.shaking,
                "tilted": self.tilted,
            },
            "gaze": {
                "eye_contact": self.eye_contact,
                "confidence": round(self.eye_contact_confidence, 2),
                "target": self.gaze_target,
            },
        }


class HeadGazeAnalyzer:
    """Analyze head orientation (3D pose angles), head gestures, and gaze contact."""

    def __init__(self) -> None:
        self.pitch_history: List[float] = []
        self.yaw_history: List[float] = []

    def analyze_frame(self, rgb_image: np.ndarray) -> GazeCues:
        """Estimate head pose and gaze direction from frame.

        Args:
            rgb_image: uint8 numpy array of shape [H, W, 3].

        Returns:
            GazeCues dataclass.
        """
        if rgb_image is None or rgb_image.size == 0:
            return GazeCues()

        h, w, c = rgb_image.shape
        if h < 20 or w < 20:
            return GazeCues()

        gray = np.mean(rgb_image, axis=-1).astype(np.float32)

        # Upper third: face and eyes region
        face_roi = gray[: int(h * 0.5), int(w * 0.2) : int(w * 0.8)]
        if face_roi.size == 0:
            return GazeCues()

        # Horizontal symmetry for Yaw (left/right rotation)
        mid_x = face_roi.shape[1] // 2
        left_half = face_roi[:, :mid_x]
        right_half = np.fliplr(face_roi[:, mid_x:])
        min_w = min(left_half.shape[1], right_half.shape[1])

        l_mean = np.mean(left_half[:, :min_w])
        r_mean = np.mean(right_half[:, :min_w])
        yaw_asym = (r_mean - l_mean) / (l_mean + r_mean + 1e-4)
        yaw_deg = float(np.clip(yaw_asym * 35.0, -45.0, 45.0))

        # Vertical mass distribution for Pitch (up/down tilt)
        mid_y = face_roi.shape[0] // 2
        top_y = np.mean(face_roi[:mid_y, :])
        bot_y = np.mean(face_roi[mid_y:, :])
        pitch_asym = (top_y - bot_y) / (top_y + bot_y + 1e-4)
        pitch_deg = float(np.clip(pitch_asym * 28.0, -35.0, 35.0))

        # Roll estimation from horizontal gradient tilt
        grad_y = np.gradient(face_roi, axis=0)
        grad_x = np.gradient(face_roi, axis=1)
        roll_angle = float(np.arctan2(np.mean(grad_y), np.mean(np.abs(grad_x)) + 1e-4) * (180.0 / np.pi))
        roll_deg = float(np.clip(roll_angle * 0.4, -30.0, 30.0))

        # Gesture detection from temporal history
        self.pitch_history.append(pitch_deg)
        self.yaw_history.append(yaw_deg)
        if len(self.pitch_history) > 12:
            self.pitch_history.pop(0)
            self.yaw_history.pop(0)

        nodding = False
        shaking = False
        if len(self.pitch_history) >= 6:
            p_arr = np.array(self.pitch_history)
            p_diffs = np.diff(p_arr)
            # Nodding shows alternating pitch velocity with significant amplitude
            sign_changes = np.sum(np.diff(np.sign(p_diffs)) != 0)
            if sign_changes >= 2 and np.ptp(p_arr) > 4.0:
                nodding = True

            y_arr = np.array(self.yaw_history)
            y_diffs = np.diff(y_arr)
            y_sign_changes = np.sum(np.diff(np.sign(y_diffs)) != 0)
            if y_sign_changes >= 2 and np.ptp(y_arr) > 6.0:
                shaking = True

        tilted = abs(roll_deg) > 7.5

        # Gaze & eye contact classification
        if abs(yaw_deg) < 14.0 and -10.0 <= pitch_deg <= 12.0:
            eye_contact = "direct"
            confidence = 0.88
            target = "camera / screen"
        elif pitch_deg < -10.0:
            eye_contact = "looking_down"
            confidence = 0.82
            target = "notes / keyboard"
        else:
            eye_contact = "looking_away"
            confidence = 0.79
            target = "off-screen / room"

        return GazeCues(
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=roll_deg,
            nodding=nodding,
            shaking=shaking,
            tilted=tilted,
            eye_contact=eye_contact,
            eye_contact_confidence=confidence,
            gaze_target=target,
        )
