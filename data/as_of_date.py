"""
As-of-date helpers — leak-prevention chokepoint.

Every cache key, every parser, every API call routes its date through here.
The invariant: nothing returned for game date X may use data dated > X.
"""

from datetime import datetime
from typing import Optional


def today_iso() -> str:
    """ISO date for 'now' (model's default as_of when none specified)."""
    return datetime.now().strftime("%Y-%m-%d")


def clamp(date: str, as_of: str) -> str:
    """Return min(date, as_of) — used to cap any stat window's right edge."""
    return min(date, as_of)


def is_future(date: str, as_of: str) -> bool:
    """True iff `date` is strictly after the as-of date (a leak)."""
    return date > as_of


def assert_no_leak(date: str, as_of: Optional[str], context: str = "") -> None:
    """Raise if a row's date leaks past the as-of cutoff. No-op if as_of is None."""
    if as_of is None:
        return
    if is_future(date, as_of):
        raise AssertionError(
            f"As-of-date leak: {context} returned date={date} "
            f"but as_of={as_of}"
        )
