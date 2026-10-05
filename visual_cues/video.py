"""Video frame extraction and temporal sampling utilities."""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

try:
    import av
except ImportError:
    av = None

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """A sampled video frame with timestamp."""

    timestamp_s: float
    frame_index: int
    image: np.ndarray  # RGB uint8 array [H, W, 3]
    width: int
    height: int


@dataclass
class VideoMetadata:
    """Metadata for a video file."""

    duration_s: float
    fps: float
    width: int
    height: int
    total_frames: int


def get_video_metadata(video_path: Union[str, Path]) -> VideoMetadata:
    """Extract metadata (duration, FPS, resolution) from a video file."""
    p = Path(video_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")

    if av is not None:
        try:
            with av.open(str(p)) as container:
                stream = container.streams.video[0]
                fps = float(stream.average_rate) if stream.average_rate else 30.0
                width = int(stream.width)
                height = int(stream.height)
                duration = float(container.duration / av.time_base) if container.duration else 0.0
                frames = int(stream.frames) if stream.frames else int(duration * fps)
                return VideoMetadata(
                    duration_s=duration,
                    fps=fps,
                    width=width,
                    height=height,
                    total_frames=frames,
                )
        except Exception as e:
            logger.debug(f"PyAV metadata failed: {e}; falling back to ffprobe")

    # Fallback to ffprobe
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,duration,nb_frames",
        "-of", "csv=p=0",
        str(p),
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip().split(",")
        width = int(out[0])
        height = int(out[1])
        r_rate = out[2].split("/")
        fps = float(r_rate[0]) / float(r_rate[1]) if len(r_rate) == 2 else float(r_rate[0])
        duration = float(out[3]) if len(out) > 3 and out[3] != "N/A" else 0.0
        frames = int(out[4]) if len(out) > 4 and out[4] != "N/A" else int(duration * fps)
        return VideoMetadata(
            duration_s=duration,
            fps=fps,
            width=width,
            height=height,
            total_frames=frames,
        )
    except Exception as e:
        logger.warning(f"Metadata probe failed: {e}; defaulting to 30fps.")
        return VideoMetadata(duration_s=0.0, fps=30.0, width=640, height=480, total_frames=0)


def iter_frames(
    video_path: Union[str, Path],
    sample_fps: float = 2.0,
    max_duration_s: Optional[float] = None,
) -> Generator[VideoFrame, None, None]:
    """Yield sampled video frames one at a time at a fixed rate (default 2.0 fps).

    This streaming generator decodes frames lazily with O(1) memory overhead,
    preventing high memory consumption on long video recordings.

    Args:
        video_path: Path to video file.
        sample_fps: Rate at which to sample frames (default: 2.0 frames/sec).
        max_duration_s: Optional cap on video duration to process.

    Yields:
        VideoFrame objects with timestamps and RGB numpy arrays.
    """
    p = Path(video_path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")

    interval_s = 1.0 / max(sample_fps, 0.1)

    if av is not None:
        try:
            with av.open(str(p)) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                last_sampled_t = -1e9
                frame_idx = 0

                for packet in container.demux(stream):
                    for av_frame in packet.decode():
                        pts_s = float(av_frame.pts * av_frame.time_base) if av_frame.pts is not None else frame_idx / 30.0
                        if max_duration_s and pts_s > max_duration_s:
                            return

                        if pts_s - last_sampled_t >= interval_s * 0.95:
                            img = av_frame.to_rgb().to_ndarray()
                            yield VideoFrame(
                                timestamp_s=round(pts_s, 3),
                                frame_index=frame_idx,
                                image=img,
                                width=av_frame.width,
                                height=av_frame.height,
                            )
                            last_sampled_t = pts_s

                        frame_idx += 1
            return
        except Exception as e:
            logger.debug(f"PyAV sampling failed: {e}; falling back to ffmpeg.")

    # Fallback using ffmpeg pipe
    cmd = [
        "ffmpeg",
        "-i", str(p),
        "-vf", f"fps={sample_fps}",
        "-f", "image2pipe",
        "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo",
        "-",
    ]
    meta = get_video_metadata(p)
    w, h = meta.width or 640, meta.height or 480
    frame_bytes = w * h * 3

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    t = 0.0
    idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            img = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            yield VideoFrame(
                timestamp_s=round(t, 3),
                frame_index=idx,
                image=img,
                width=w,
                height=h,
            )
            t += interval_s
            idx += 1
            if max_duration_s and t > max_duration_s:
                break
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait()


def sample_frames(
    video_path: Union[str, Path],
    sample_fps: float = 2.0,
    max_duration_s: Optional[float] = None,
) -> List[VideoFrame]:
    """Sample video frames into a list at a fixed rate (default 2.0 fps).

    Note: For long recordings (>5 minutes), prefer iter_frames() or analyze_video() to avoid
    high memory accumulation.

    Args:
        video_path: Path to video file.
        sample_fps: Rate at which to sample frames (default: 2.0 frames/sec).
        max_duration_s: Optional cap on video duration to process.

    Returns:
        List of VideoFrame objects with timestamps and RGB numpy arrays.
    """
    return list(iter_frames(video_path, sample_fps=sample_fps, max_duration_s=max_duration_s))
