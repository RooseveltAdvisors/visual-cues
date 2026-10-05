"""Unit tests for FaceAnalyzer in visual_cues."""

import numpy as np
import pytest

from visual_cues.face import FaceAnalyzer, FaceCues


@pytest.fixture
def analyzer():
    return FaceAnalyzer()


def test_empty_image(analyzer):
    res = analyzer.analyze_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert not res.face_detected
    d = res.to_dict()
    assert d["face_detected"] is False


def test_synthetic_face_neutral(analyzer):
    # Create synthetic frame with skin tone oval in center
    h, w = 240, 320
    img = np.ones((h, w, 3), dtype=np.uint8) * 80  # dark background
    # Skin tone rectangle (R > G > B)
    img[60:160, 100:220] = [180, 130, 95]

    res = analyzer.analyze_frame(img)
    assert res.face_detected
    assert res.confidence > 0.5
    assert -0.8 <= res.valence <= 0.8
    assert "AU06_12_smile" in res.to_dict()["action_units"]
    assert "AU04_brow_furrow" in res.to_dict()["action_units"]


def test_face_cues_serialization(analyzer):
    img = np.ones((100, 100, 3), dtype=np.uint8) * 150
    cues = analyzer.analyze_frame(img)
    d = cues.to_dict()
    assert "face_detected" in d
    assert "valence" in d
    assert "action_units" in d
    assert "expression_scores" in d
