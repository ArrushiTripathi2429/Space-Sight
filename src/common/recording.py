from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class FrameRecorder:
    def __init__(self, root: str = "data/processed"):
        started_at = datetime.now(timezone.utc)
        session_id = started_at.strftime("session_%Y%m%d_%H%M%S_%f")
        self.session_dir = Path(root) / session_id
        self.session_dir.mkdir(parents=True, exist_ok=False)
        self.frames_path = self.session_dir / "frames.jsonl"
        self.index_path = self.session_dir / "index.json"
        self.metadata_path = self.session_dir / "metadata.json"
        self._frames_file = self.frames_path.open("w", encoding="utf-8")
        self._index = []
        self._frame_count = 0
        self._first_frame_id = None
        self._last_frame_id = None
        self._first_timestamp_ms = None
        self._last_timestamp_ms = None

        self._write_json(
            self.metadata_path,
            {
                "schema_version": "1.0",
                "session_id": session_id,
                "started_at": started_at.isoformat(),
                "record_format": "newline-delimited JSON",
                "frames_file": "frames.jsonl",
                "index_file": "index.json",
                "fields": [
                    "frame_id",
                    "timestamp_ms",
                    "objects",
                    "pose",
                    "hands",
                    "interactions",
                ],
            },
        )

    def write(self, frame_data: Dict[str, Any]) -> None:
        frame_id = int(frame_data["frame_id"])
        timestamp_ms = int(frame_data["timestamp_ms"])
        byte_offset = self._frames_file.tell()
        self._frames_file.write(json.dumps(frame_data, separators=(",", ":")) + "\n")
        self._frames_file.flush()

        self._index.append(
            {
                "frame_id": frame_id,
                "timestamp_ms": timestamp_ms,
                "line": self._frame_count,
                "byte_offset": byte_offset,
            }
        )
        self._frame_count += 1
        self._first_frame_id = frame_id if self._first_frame_id is None else self._first_frame_id
        self._last_frame_id = frame_id
        self._first_timestamp_ms = (
            timestamp_ms
            if self._first_timestamp_ms is None
            else self._first_timestamp_ms
        )
        self._last_timestamp_ms = timestamp_ms

    def close(self) -> None:
        if self._frames_file.closed:
            return

        self._frames_file.close()
        self._write_json(self.index_path, self._index)
        self._write_json(
            self.metadata_path,
            {
                "schema_version": "1.0",
                "session_id": self.session_dir.name,
                "record_format": "newline-delimited JSON",
                "frames_file": "frames.jsonl",
                "index_file": "index.json",
                "frame_count": self._frame_count,
                "first_frame_id": self._first_frame_id,
                "last_frame_id": self._last_frame_id,
                "first_timestamp_ms": self._first_timestamp_ms,
                "last_timestamp_ms": self._last_timestamp_ms,
                "fields": [
                    "frame_id",
                    "timestamp_ms",
                    "objects",
                    "pose",
                    "hands",
                    "interactions",
                ],
            },
        )

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(value, indent=2),
            encoding="utf-8",
        )