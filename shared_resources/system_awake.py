"""Cross-platform keep-awake helper for long-running OCT jobs."""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
from contextlib import contextmanager
from typing import Iterator, Optional


@contextmanager
def keep_system_awake(enabled: bool = True, reason: Optional[str] = None) -> Iterator[None]:
    """Prevent the current machine from sleeping while inside the block.

    macOS uses ``caffeinate``; Windows uses ``SetThreadExecutionState``.
    Other platforms fall back to a no-op.
    """
    if not enabled:
        yield
        return

    system_name = platform.system().lower()
    label = f" for {reason}" if reason else ""

    if system_name == 'darwin':
        process = None
        try:
            process = subprocess.Popen(
                ['caffeinate', '-dimsu', '-w', str(os.getpid())],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print(f"  Keep-awake enabled via caffeinate{label}")
            yield
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
        return

    if system_name == 'windows':
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_DISPLAY_REQUIRED = 0x00000002
        try:
            kernel32 = ctypes.windll.kernel32
            prev = kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
            )
            if prev == 0:
                print(f"  WARNING: unable to enable Windows keep-awake{label}")
            else:
                print(f"  Keep-awake enabled via SetThreadExecutionState{label}")
            yield
        finally:
            try:
                kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass
        return

    print(f"  Keep-awake not enabled on platform '{system_name}'{label}")
    yield