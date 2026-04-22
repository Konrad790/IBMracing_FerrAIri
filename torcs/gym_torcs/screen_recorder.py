from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ScreenRecorderConfig:
    ffmpeg_path: str = "ffmpeg"
    fps: int = 30
    capture_source: str = "desktop"
    preset: str = "ultrafast"
    crf: int = 23


class FfmpegScreenRecorder:
    def __init__(self, config: ScreenRecorderConfig | None = None) -> None:
        self.config = config or ScreenRecorderConfig()
        self._process: subprocess.Popen[bytes] | None = None
        self._output_path: Path | None = None

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    def is_available(self) -> bool:
        return shutil.which(self.config.ffmpeg_path) is not None

    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, output_path: str | Path) -> Path:
        if not self.is_available():
            raise RuntimeError(
                "ffmpeg is not installed or not available on PATH. "
                "Install ffmpeg to enable automatic lap video recording."
            )
        if self.is_recording():
            raise RuntimeError("Screen recording is already active.")

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.config.ffmpeg_path,
            "-y",
            "-f",
            "gdigrab",
            "-framerate",
            str(self.config.fps),
            "-i",
            self.config.capture_source,
            "-c:v",
            "libx264",
            "-preset",
            self.config.preset,
            "-crf",
            str(self.config.crf),
            "-pix_fmt",
            "yuv420p",
            str(path),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._output_path = path
        return path

    def stop(self, *, keep_file: bool = True, timeout_seconds: float = 10.0) -> Path | None:
        path = self._output_path
        process = self._process
        self._process = None
        self._output_path = None

        if process is not None and process.poll() is None:
            try:
                if process.stdin is not None:
                    process.stdin.write(b"q\n")
                    process.stdin.flush()
                process.wait(timeout=timeout_seconds)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=3.0)
                except Exception:
                    process.kill()
                    process.wait(timeout=3.0)

        if not keep_file and path is not None and path.exists():
            path.unlink(missing_ok=True)
            return None

        return path
