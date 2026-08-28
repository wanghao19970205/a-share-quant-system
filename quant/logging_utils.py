"""Small line-oriented stdout helpers for scheduled job logs."""
from __future__ import annotations

import datetime as dt
import re
import sys
from typing import TextIO

_TIMESTAMP_PREFIX = re.compile(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\]")


class TimestampedWriter:
    """Prefix complete stdout lines without changing print/flush semantics."""

    def __init__(self, stream: TextIO):
        self.stream = stream
        self.buffer = ""

    def write(self, data: str) -> int:
        if not data:
            return 0
        self.buffer += data
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._write_line(line)
        return len(data)

    def _write_line(self, line: str) -> None:
        prefix = "" if _TIMESTAMP_PREFIX.match(line) else f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] "
        self.stream.write(prefix + line + "\n")

    def flush(self) -> None:
        if self.buffer:
            self._write_line(self.buffer)
            self.buffer = ""
        self.stream.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    def fileno(self) -> int:
        return self.stream.fileno()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def install_timestamped_stdout() -> None:
    """Install once per process; keep stderr untouched for tracebacks/progress bars."""
    if not isinstance(sys.stdout, TimestampedWriter):
        sys.stdout = TimestampedWriter(sys.stdout)
