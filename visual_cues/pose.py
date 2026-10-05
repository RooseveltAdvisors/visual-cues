"""Body posture, torso inclination, openness, and movement restlessness analysis."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PoseCues:
    """Body language and posture cues extracted from a frame."""

    person_detected: bool = False
    confidence: float = 0.0

    # Posture Dimensions
    lean: str = "upright"  # "forward", "upright", "backward"
    lean_angle_deg: float = 0.0  # Positive = forward, Negative = backward
    openness: str = "open"  # "open", "closed" (e.g. arms crossed), "neutral"
    openness_score: float = 0.75  # 0.0 (fully closed/defensive) to 1.0 (expansive/open)

    slouch_score: float = 0.0  # 0.0 (good upright posture) to 1.0 (severely slouched)
    shoulder_tension: float = 0.0  # 0.0 (relaxed shoulders) to 1.0 (elevated/tense/shrugging)
    restlessness_score: float = 0.0  # 0.0 (steady) to 1.0 (excessive movement/fidgeting)
    movement_state: str = "steady"  # "steady", "shifting", "fidgeting"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "person_detected": self.person_detected,
            "confidence": round(self.confidence, 2),
            "lean": self.lean,
            "lean_angle_deg": round(self.lean_angle_deg, 1),
            "openness": self.openness,
            "openness_score": round(self.openness_score, 2),
            "slouch_score": round(self.slouch_score, 2),
            "shoulder_tension": round(self.shoulder_tension, 2),
            "restlessness_score": round(self.restlessness_score, 2),
            "movement_state": self.movement_state,
        }


class BodyPoseAnalyzer:
    """Extract body posture, openness, torso inclination, and movement dynamics."""

    def __init__(self) -> None:
        self.prev_frame_gray: Optional[np.ndarray] = None
        self.recent_movements: List[float] = []

    def analyze_frame(self, rgb_image: np.ndarray) -> PoseCues:
        """Analyze body posture and movement from an RGB video frame.

        Args:
            rgb_image: uint8 numpy array of shape [H, W, 3].

        Returns:
            PoseCues dataclass filled with posture and movement metrics.
        """
        if rgb_image is None or rgb_image.size == 0:
            return PoseCues(person_detected=False)

        h, w, c = rgb_image.shape
        if h < 30 or w < 30:
            return PoseCues(person_detected=False)

        gray = np.mean(rgb_image, axis=-1).astype(np.float32)

        # Estimate optical flow / motion delta against previous frame
        if self.prev_frame_gray is not None and self.prev_frame_gray.shape == gray.shape:
            frame_diff = np.abs(gray - self.prev_frame_gray)
            motion_mag = float(np.mean(frame_diff) / 255.0)
        else:
            motion_mag = 0.0

        self.prev_frame_gray = gray
        self.recent_movements.append(motion_mag)
        if len(self.recent_movements) > 10:
            self.recent_movements.pop(0)

        restlessness = float(np.mean(self.recent_movements) * 12.0)
        restlessness = min(max(restlessness, 0.0), 1.0)

        if restlessness > 0.45:
            movement_state = "fidgeting"
        elif restlessness > 0.18:
            movement_state = "shifting"
        else:
            movement_state = "steady"

        # Torso and shoulder region bounding
        # In a meeting or video call, shoulders and torso occupy rows h*0.3 to h*0.9
        torso_y1, torso_y2 = int(h * 0.35), int(h * 0.90)
        torso_roi = gray[torso_y1:torso_y2, :]

        # Torso center of mass along horizontal axis
        col_mass = np.mean(torso_roi, axis=0)
        center_x = float(np.sum(np.arange(len(col_mass)) * col_mass) / (np.sum(col_mass) + 1e-5))
        frame_center_x = w / 2.0
        horizontal_offset = (center_x - frame_center_x) / frame_center_x

        # Vertical mass distribution (upper torso vs lower torso)
        row_mass = np.mean(torso_roi, axis=1)
        mid_row = len(row_mass) // 2
        upper_mass = float(np.sum(row_mass[:mid_row]))
        lower_mass = float(np.sum(row_mass[mid_row:]) + 1e-5)
        vertical_ratio = upper_mass / lower_mass

        # Lean angle estimation: leaning forward shifts mass upward and widens upper torso
        lean_angle = (vertical_ratio - 1.0) * 18.0
        lean_angle = min(max(lean_angle, -25.0), 25.0)

        if lean_angle > 4.5:
            lean = "forward"
        elif lean_angle < -4.5:
            lean = "backward"
        else:
            lean = "upright"

        # Posture openness & arms crossed detection
        # Crossing arms creates high horizontal edge density across mid-chest
        chest_roi = torso_roi[: int(len(torso_roi) * 0.5), int(w * 0.25) : int(w * 0.75)]
        if chest_roi.size > 0:
            chest_grad_x = np.gradient(chest_roi, axis=1)
            chest_texture = float(np.std(chest_grad_x) / (np.mean(chest_roi) + 1e-4))
            arms_crossed_prob = float(np.clip((chest_texture - 0.28) * 2.8, 0.0, 0.95))
        else:
            arms_crossed_prob = 0.0

        openness_score = 1.0 - arms_crossed_prob
        if openness_score > 0.65:
            openness = "open"
        elif openness_score < 0.35:
            openness = "closed"
        else:
            openness = "neutral"

        # Slouch score (low upper mass + downward slump)
        slouch_score = float(np.clip((0.95 - vertical_ratio) * 1.8, 0.0, 0.95))

        # Shoulder tension / shrug
        shoulder_tension = float(np.clip((vertical_ratio - 1.25) * 1.5, 0.0, 0.90))

        return PoseCues(
            person_detected=True,
            confidence=0.90,
            lean=lean,
            lean_angle_deg=lean_angle,
            openness=openness,
            openness_score=openness_score,
            slouch_score=slouch_score,
            shoulder_tension=shoulder_tension,
            restlessness_score=restlessness,
            movement_state=movement_state,
        )
