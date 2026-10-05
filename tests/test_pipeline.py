"""Unit and integration tests for VisualCuesPipeline."""

import numpy as np
import pytest

from visual_cues.pipeline import (
    TurnVisualSummary,
    VisualCuesPipeline,
    VisualCuesResult,
)
from visual_cues.video import VideoFrame


def test_pipeline_turn_aggregation():
    pipeline = VisualCuesPipeline()

    # Create synthetic frames spanning 0.0s to 4.0s
    frames = []
    for idx, t in enumerate([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]):
        img = np.ones((120, 160, 3), dtype=np.uint8) * 120
        # Add face-like blob
        img[30:70, 60:100] = [190, 140, 100]
        frames.append(VideoFrame(timestamp_s=t, frame_index=idx, image=img, width=160, height=120))

    turn_intervals = [(0.0, 2.0), (2.0, 4.0)]
    res = pipeline.process_frames(frames, video_path="mock.mp4", turn_intervals=turn_intervals)

    assert isinstance(res, VisualCuesResult)
    assert len(res.timeline) == 7
    assert len(res.turn_summaries) == 2

    # Check turn summary fields
    t0 = res.turn_summaries[0]
    assert isinstance(t0, TurnVisualSummary)
    assert t0.start == 0.0
    assert t0.end == 2.0
    assert t0.frames_count > 0
    assert t0.posture_lean in ("forward", "upright", "backward")
    assert 0.0 <= t0.engagement_score <= 1.0

    # JSON serialization
    json_dict = res.to_dict()
    assert "video_summary" in json_dict
    assert "turn_summaries" in json_dict
    assert len(json_dict["turn_summaries"]) == 2
    assert "timeline" in json_dict


def test_pipeline_streaming_generator():
    """Verify that process_frames accepts an arbitrary generator/iterator with O(1) storage."""
    pipeline = VisualCuesPipeline()

    def frame_gen():
        for idx, t in enumerate([1.0, 2.0, 3.0]):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            yield VideoFrame(timestamp_s=t, frame_index=idx, image=img, width=100, height=100)

    res = pipeline.process_frames(frame_gen(), video_path="streaming_mock.mp4")
    assert isinstance(res, VisualCuesResult)
    assert len(res.timeline) == 3
    assert res.duration_s == 3.0
    assert res.total_frames_analyzed == 3

