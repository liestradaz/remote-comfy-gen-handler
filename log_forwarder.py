"""Buffered logger that forwards logs to avivs_logger service.

Sends structured logs with source, tags, and levels to
https://logging.avivkaplan.com/ingest. Buffers lines and flushes
periodically to reduce HTTP overhead.

Also prints all lines to stderr so they appear in RunPod's built-in logs.

Environment variables:
    AVIVS_LOGGER_TOKEN  - Bearer token for auth (required for forwarding)
    AVIVS_LOGGER_URL    - Override endpoint (default: https://logging.avivkaplan.com)
    RUNPOD_ENDPOINT_ID  - Used as tag
    RUNPOD_POD_ID       - Used as tag (worker_id)
"""
from __future__ import annotations

import json
import os
import sys
import threading
import urllib.request

_LOGGER_URL = os.environ.get("AVIVS_LOGGER_URL", os.environ.get("LOG_RECEIVER_URL", "https://logging.avivkaplan.com"))
_LOGGER_TOKEN = os.environ.get("AVIVS_LOGGER_TOKEN", os.environ.get("LOG_RECEIVER_TOKEN", ""))

SOURCE = "comfy-gen-worker"


class AvivLogger:
    def __init__(self, source: str = SOURCE, tags: dict[str, str] | None = None,
                 flush_interval: float = 5.0):
        self.source = source
        self.tags = tags or {}
        self._url = _LOGGER_URL.rstrip("/") + "/ingest"
        self._token = _LOGGER_TOKEN
        self._enabled = bool(self._token)
        self._buffer: list[tuple[str, str]] = []  # (level, line)
        self._lock = threading.Lock()
        self._flush_interval = flush_interval
        self._timer: threading.Timer | None = None
        if self._enabled:
            self._schedule_flush()

    def _schedule_flush(self) -> None:
        self._timer = threading.Timer(self._flush_interval, self._auto_flush)
        self._timer.daemon = True
        self._timer.start()

    def _auto_flush(self) -> None:
        self.flush()
        self._schedule_flush()

    def info(self, line: str, **extra_tags: str) -> None:
        self._append("INFO", line, extra_tags)

    def warn(self, line: str, **extra_tags: str) -> None:
        self._append("WARN", line, extra_tags)

    def error(self, line: str, **extra_tags: str) -> None:
        self._append("ERROR", line, extra_tags)

    def _append(self, level: str, line: str, extra_tags: dict[str, str] | None = None) -> None:
        # Always print to stderr for RunPod's built-in log viewer
        print(line, file=sys.stderr, flush=True)
        if not self._enabled:
            return
        with self._lock:
            self._buffer.append((level, line))

    def flush(self) -> None:
        with self._lock:
            if not self._buffer:
                return
            batches: dict[str, list[str]] = {}
            for level, line in self._buffer:
                batches.setdefault(level, []).append(line)
            self._buffer.clear()

        for level, lines in batches.items():
            payload = json.dumps({
                "source": self.source,
                "tags": self.tags,
                "level": level,
                "lines": lines,
            }).encode()
            req = urllib.request.Request(
                self._url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._token}",
                },
            )
            try:
                urllib.request.urlopen(req, timeout=5)
            except Exception:
                pass  # never let logging failures break the worker

    def with_tags(self, **tags: str) -> AvivLogger:
        """Return a new logger with additional tags merged in."""
        merged = {**self.tags, **tags}
        child = AvivLogger(source=self.source, tags=merged,
                           flush_interval=self._flush_interval)
        # Share the parent's buffer and token state
        child._enabled = self._enabled
        child._token = self._token
        child._url = self._url
        return child


# Module-level default logger instance
_default_tags = {}
_endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", os.environ.get("RUNPOD_DC_ID", ""))
_worker_id = os.environ.get("RUNPOD_POD_ID", os.environ.get("HOSTNAME", ""))
if _endpoint_id:
    _default_tags["endpoint"] = _endpoint_id
if _worker_id:
    _default_tags["worker"] = _worker_id

log = AvivLogger(source=SOURCE, tags=_default_tags)
