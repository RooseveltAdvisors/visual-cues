"""Unit tests for BodyPoseAnalyzer in visual_cues."""

import numpy as np
import pytest

from visual_cues.pose import BodyPoseAnalyzer, PoseCues


@pytest.fixture
def analyzer():
    return BodyPoseAnalyzer()


def test_empty_frame(analyzer):
    res = analyzer.analyze_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not res.person_detected


def test_upright_neutral_pose(analyzer):
    h, w = 240, 320
    img = np.ones((h, w, 3), dtype=np.uint8) * 50
    # Add body mass in center
    img[80:220, 100:220] = 160

    res = analyzer.analyze_frame(img)
    assert res.person_detected
    assert res.lean in ("forward", "upright", "backward")
    assert res.openness in ("open", "closed", "neutral")
    assert 0.0 <= res.slouch_score <= 1.0
    assert 0.0 <= res.restlessness_score <= 1.0


def test_movement_detection(analyzer):
    h, w = 100, 100
    frame1 = np.ones((h, w, 3), dtype=np.uint8) * 50
    frame2 = np.ones((h, w, 3), dtype=np.uint8) * 200  # large shift

    _ = analyzer.analyze_frame(frame1)
    res2 = analyzer.analyze_frame(frame2)
    assert res2.restlessness_score > 0.0
