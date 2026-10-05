"""Command line interface for visual-cues."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from visual_cues.pipeline import VisualCuesPipeline

logger = logging.getLogger(__name__)


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="visual-cues",
        description="Extract facial expressions, FACS Action Units, head pose, gaze, and body language from video recordings",
    )
    parser.add_argument(
        "--video",
        "-v",
        type=str,
        required=True,
        help="Path to input video recording (MP4, MKV, MOV, WebM, etc.)",
    )
    parser.add_argument(
        "--out",
        "-o",
        type=str,
        default=None,
        help="Path to write text summary output",
    )
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        nargs="?",
        const="",
        default=None,
        help="Path to write structured JSON output (or prints to stdout if flag provided with no path)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Sampling rate in frames per second (default: 2.0)",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=None,
        help="Optional maximum seconds of video to process",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)

    log_level = logging.DEBUG if parsed.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    video_p = Path(parsed.video).expanduser()
    if not video_p.exists():
        sys.stderr.write(f"Error: video file '{video_p}' not found.\n")
        return 1

    try:
        pipeline = VisualCuesPipeline()
        result = pipeline.analyze_video(
            video_path=video_p,
            sample_fps=parsed.fps,
            max_duration_s=parsed.max_duration,
        )

        # Summary text formatting
        lines = [
            f"=== Visual Cues Analysis: {video_p.name} ===",
            f"Duration: {result.duration_s:.1f}s | Analyzed Frames: {result.total_frames_analyzed}",
        ]
        if result.video_summary:
            vs = result.video_summary
            lines.extend([
                f"Overall Affect Valence: {vs.get('avg_valence', 0.0):+.2f}",
                f"Eye Contact Ratio: {vs.get('eye_contact_ratio', 0.0)*100:.1f}%",
                f"Open Posture Ratio: {vs.get('open_posture_ratio', 0.0)*100:.1f}%",
                f"Smile Frequency: {vs.get('smile_frequency', 0.0)*100:.1f}%",
                f"Restlessness Index: {vs.get('restlessness_index', 0.0):.2f}",
                f"Engagement Rating: {vs.get('engagement_rating', 'moderate').upper()}",
            ])

        summary_text = "\n".join(lines) + "\n"

        if parsed.out:
            out_p = Path(parsed.out).expanduser()
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(summary_text, encoding="utf-8")
            logging.info(f"Visual cues summary written to {out_p}")

        if parsed.json is not None:
            json_str = result.to_json()
            if parsed.json != "":
                json_p = Path(parsed.json).expanduser()
                json_p.parent.mkdir(parents=True, exist_ok=True)
                json_p.write_text(json_str, encoding="utf-8")
                logging.info(f"Structured visual JSON written to {json_p}")
            else:
                print(json_str)

        if not parsed.out and parsed.json is None:
            print(summary_text)

        return 0

    except Exception as e:
        logging.exception(f"Visual cues analysis failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
