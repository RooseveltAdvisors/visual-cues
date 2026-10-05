"""End-to-end visual cue extraction pipeline for video recordings."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from visual_cues.face import FaceAnalyzer, FaceCues
from visual_cues.gaze import GazeCues, HeadGazeAnalyzer
from visual_cues.pose import BodyPoseAnalyzer, PoseCues
from visual_cues.video import VideoFrame, get_video_metadata, sample_frames

logger = logging.getLogger(__name__)


@dataclass
class FrameVisualCues:
    """Consolidated visual cues for a single timestamp."""

    timestamp_s: float
    frame_index: int
    face: FaceCues
    pose: PoseCues
    gaze: GazeCues

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_s": round(self.timestamp_s, 2),
            "frame_index": self.frame_index,
            "face": self.face.to_dict(),
            "pose": self.pose.to_dict(),
            "gaze": self.gaze.to_dict(),
        }


@dataclass
class TurnVisualSummary:
    """Aggregated visual cues over a specific turn interval [start, end]."""

    start: float
    end: float
    frames_count: int
    dominant_expression: str = "neutral"
    avg_valence: float = 0.0
    smile_intensity: float = 0.0
    brow_furrow: float = 0.0
    posture_lean: str = "upright"
    posture_openness: str = "open"
    eye_contact_ratio: float = 1.0
    head_nodding_detected: bool = False
    head_shaking_detected: bool = False
    restlessness_score: float = 0.0
    engagement_score: float = 0.80  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "frames_count": self.frames_count,
            "dominant_expression": self.dominant_expression,
            "avg_valence": round(self.avg_valence, 2),
            "smile_intensity": round(self.smile_intensity, 2),
            "brow_furrow": round(self.brow_furrow, 2),
            "posture_lean": self.posture_lean,
            "posture_openness": self.posture_openness,
            "eye_contact_ratio": round(self.eye_contact_ratio, 2),
            "nodding": self.head_nodding_detected,
            "shaking": self.head_shaking_detected,
            "restlessness": round(self.restlessness_score, 2),
            "engagement_score": round(self.engagement_score, 2),
        }


@dataclass
class VisualCuesResult:
    """Complete structured visual cues analysis for a video."""

    video_path: str
    duration_s: float
    total_frames_analyzed: int
    timeline: List[FrameVisualCues] = field(default_factory=list)
    turn_summaries: List[TurnVisualSummary] = field(default_factory=list)
    video_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "duration_s": round(self.duration_s, 2),
            "frames_analyzed": self.total_frames_analyzed,
            "video_summary": self.video_summary,
            "turn_summaries": [ts.to_dict() for ts in self.turn_summaries],
            "timeline": [f.to_dict() for f in self.timeline],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


class VisualCuesPipeline:
    """Process video recordings to extract facial affect, head pose, gaze, and body language."""

    def __init__(
        self,
        face_analyzer: Optional[FaceAnalyzer] = None,
        pose_analyzer: Optional[BodyPoseAnalyzer] = None,
        gaze_analyzer: Optional[HeadGazeAnalyzer] = None,
    ) -> None:
        self.face_analyzer = face_analyzer or FaceAnalyzer()
        self.pose_analyzer = pose_analyzer or BodyPoseAnalyzer()
        self.gaze_analyzer = gaze_analyzer or HeadGazeAnalyzer()

    def process_frames(
        self,
        frames: List[VideoFrame],
        video_path: str = "video",
        turn_intervals: Optional[List[Tuple[float, float]]] = None,
    ) -> VisualCuesResult:
        """Run visual cues analysis over a pre-sampled list of video frames."""
        timeline: List[FrameVisualCues] = []

        for vf in frames:
            face_cues = self.face_analyzer.analyze_frame(vf.image)
            pose_cues = self.pose_analyzer.analyze_frame(vf.image)
            gaze_cues = self.gaze_analyzer.analyze_frame(vf.image)

            timeline.append(
                FrameVisualCues(
                    timestamp_s=vf.timestamp_s,
                    frame_index=vf.frame_index,
                    face=face_cues,
                    pose=pose_cues,
                    gaze=gaze_cues,
                )
            )

        duration = max([f.timestamp_s for f in frames], default=0.0)

        # Compute turn-aligned summaries if intervals provided
        turn_summaries: List[TurnVisualSummary] = []
        if turn_intervals:
            for s, e in turn_intervals:
                # Find frames within [s - 0.2, e + 0.2]
                turn_frames = [f for f in timeline if (s - 0.25) <= f.timestamp_s <= (e + 0.25)]
                if not turn_frames:
                    # Nearest frame
                    turn_frames = min(timeline, key=lambda f: abs(f.timestamp_s - (s + e) / 2.0), default=None)
                    turn_frames = [turn_frames] if turn_frames else []

                if turn_frames:
                    valences = [f.face.valence for f in turn_frames]
                    smiles = [f.face.au_smile for f in turn_frames]
                    furrows = [f.face.au_brow_furrow for f in turn_frames]
                    leans = [f.pose.lean for f in turn_frames]
                    opennesses = [f.pose.openness for f in turn_frames]
                    eye_contacts = [1.0 if f.gaze.eye_contact == "direct" else 0.0 for f in turn_frames]
                    nods = any(f.gaze.nodding for f in turn_frames)
                    shakes = any(f.gaze.shaking for f in turn_frames)
                    restlessness = float(np.mean([f.pose.restlessness_score for f in turn_frames]))
                    expressions = [f.face.dominant_expression for f in turn_frames]

                    dom_exp = max(set(expressions), key=expressions.count) if expressions else "neutral"
                    dom_lean = max(set(leans), key=leans.count) if leans else "upright"
                    dom_open = max(set(opennesses), key=opennesses.count) if opennesses else "open"
                    ec_ratio = float(np.mean(eye_contacts)) if eye_contacts else 0.85

                    # Engagement score derived from eye contact + open posture + attentive lean
                    lean_bonus = 0.15 if dom_lean == "forward" else (-0.1 if dom_lean == "backward" else 0.05)
                    open_bonus = 0.15 if dom_open == "open" else -0.15
                    engagement = float(np.clip(0.6 * ec_ratio + lean_bonus + open_bonus + (0.1 if nods else 0.0), 0.1, 0.98))

                    turn_summaries.append(
                        TurnVisualSummary(
                            start=s,
                            end=e,
                            frames_count=len(turn_frames),
                            dominant_expression=dom_exp,
                            avg_valence=float(np.mean(valences)),
                            smile_intensity=float(np.mean(smiles)),
                            brow_furrow=float(np.mean(furrows)),
                            posture_lean=dom_lean,
                            posture_openness=dom_open,
                            eye_contact_ratio=ec_ratio,
                            head_nodding_detected=nods,
                            head_shaking_detected=shakes,
                            restlessness_score=restlessness,
                            engagement_score=engagement,
                        )
                    )

        # Video-level aggregated summary
        if timeline:
            all_valences = [f.face.valence for f in timeline]
            all_smiles = [f.face.au_smile for f in timeline]
            all_ec = [1.0 if f.gaze.eye_contact == "direct" else 0.0 for f in timeline]
            all_open = [1.0 if f.pose.openness == "open" else 0.0 for f in timeline]
            all_restless = [f.pose.restlessness_score for f in timeline]

            video_summary = {
                "avg_valence": round(float(np.mean(all_valences)), 2),
                "smile_frequency": round(float(np.mean([1.0 if s > 0.3 else 0.0 for s in all_smiles])), 2),
                "eye_contact_ratio": round(float(np.mean(all_ec)), 2),
                "open_posture_ratio": round(float(np.mean(all_open)), 2),
                "restlessness_index": round(float(np.mean(all_restless)), 2),
                "engagement_rating": "high" if float(np.mean(all_ec)) > 0.75 and float(np.mean(all_open)) > 0.6 else "moderate",
            }
        else:
            video_summary = {}

        return VisualCuesResult(
            video_path=str(video_path),
            duration_s=duration,
            total_frames_analyzed=len(timeline),
            timeline=timeline,
            turn_summaries=turn_summaries,
            video_summary=video_summary,
        )

    def analyze_video(
        self,
        video_path: Union[str, Path],
        sample_fps: float = 2.0,
        turn_intervals: Optional[List[Tuple[float, float]]] = None,
        max_duration_s: Optional[float] = None,
    ) -> VisualCuesResult:
        """Sample video and execute complete visual cues analysis."""
        p = Path(video_path).expanduser()
        logger.info(f"Sampling video frames at {sample_fps} FPS: {p}")
        frames = sample_frames(p, sample_fps=sample_fps, max_duration_s=max_duration_s)
        logger.info(f"Sampled {len(frames)} frames; running visual cue analyzers...")
        return self.process_frames(frames, video_path=str(p), turn_intervals=turn_intervals)
