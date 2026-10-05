"""Unit tests for visual-cues CLI."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from visual_cues.cli import main, parse_args
from visual_cues.pipeline import VisualCuesResult


def test_parse_args_defaults():
    args = parse_args(["--video", "test.mp4"])
    assert args.video == "test.mp4"
    assert args.fps == 2.0
    assert args.out is None
    assert args.json is None


def test_cli_missing_file():
    code = main(["--video", "nonexistent_file_xyz.mp4"])
    assert code == 1


@patch("visual_cues.cli.VisualCuesPipeline")
def test_cli_execution_mocked(mock_pipeline_cls, tmp_path):
    mock_pipeline = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline

    mock_result = VisualCuesResult(
        video_path="test.mp4",
        duration_s=10.0,
        total_frames_analyzed=20,
        video_summary={
            "avg_valence": 0.45,
            "eye_contact_ratio": 0.90,
            "open_posture_ratio": 0.85,
            "engagement_rating": "high",
        },
    )
    mock_pipeline.analyze_video.return_value = mock_result

    video_file = tmp_path / "mock.mp4"
    video_file.touch()
    out_file = tmp_path / "out.txt"
    json_file = tmp_path / "out.json"

    code = main([
        "--video", str(video_file),
        "--out", str(out_file),
        "--json", str(json_file),
    ])
    assert code == 0
    assert out_file.exists()
    assert "Overall Affect Valence: +0.45" in out_file.read_text()
    assert json_file.exists()
