"""Unit tests for HeadGazeAnalyzer in visual_cues."""

import numpy as np
import pytest

from visual_cues.gaze import GazeCues, HeadGazeAnalyzer


@pytest.fixture
def analyzer():
    return HeadGazeAnalyzer()


def test_empty_image(analyzer):
    res = analyzer.analyze_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert res.eye_contact in ("direct", "looking_down", "looking_away")


def test_centered_head_gaze(analyzer):
    h, w = 200, 200
    img = np.ones((h, w, 3), dtype=np.uint8) * 100
    # Symmetric face region
    img[30:100, 60:140] = 180

    res = analyzer.analyze_frame(img)
    assert abs(res.yaw_deg) < 30.0
    assert abs(res.pitch_deg) < 30.0
    assert res.eye_contact in ("direct", "looking_down", "looking_away")
    d = res.to_dict()
    assert "head_pose" in d
    assert "gaze" in d
