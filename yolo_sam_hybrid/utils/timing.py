"""Timing utilities."""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Dict


@contextmanager
def timer(times: Dict[str, float], key: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        times[key] = times.get(key, 0.0) + time.perf_counter() - start
