#!/usr/bin/env python3
"""Run Eve GUI and preserve full traceback output in a log file.

Usage:
  python eve_traceback_launcher.py

It launches `eve_terminal_gui_cosmic.py` in a subprocess, streams output to your
terminal, and mirrors stderr to `eve_crash_traceback.log`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

LOG_PATH = Path("eve_crash_traceback.log")


def main() -> int:
    cmd = [sys.executable, "eve_terminal_gui_cosmic.py"]

    with LOG_PATH.open("a", encoding="utf-8") as log_file:
        log_file.write("\n\n=== New launch session ===\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)

        return process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
