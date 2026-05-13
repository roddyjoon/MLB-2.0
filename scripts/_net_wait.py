"""
Wait for network connectivity before proceeding.

launchd fires scripts as soon as the calendar interval hits — but if the
Mac was asleep, Wi-Fi/DNS may take 30-60 seconds to come up after wake.
The morning of 2026-05-13 both jobs fired at 08:00/08:05 PT and got
`Cannot connect to host api.resend.com:443 [DNS resolution failed]`
because Wi-Fi was still negotiating.

Polling `socket.gethostbyname("api.resend.com")` is cheap and definitive:
once it succeeds, all downstream HTTPS calls will too.
"""

import socket
import sys
import time
from typing import Iterable


DEFAULT_HOSTS = (
    "api.resend.com",
    "statsapi.mlb.com",
    "api.actionnetwork.com",
)


def wait_for_network(
    hosts: Iterable[str] = DEFAULT_HOSTS,
    timeout_seconds: int = 300,
    interval_seconds: int = 10,
    verbose: bool = True,
) -> bool:
    """Poll DNS resolution until all `hosts` resolve, or timeout.

    Returns True if network came up in time, False otherwise.
    Prints status to stderr each retry so launchd logs show progress.
    """
    deadline = time.time() + timeout_seconds
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            for h in hosts:
                socket.gethostbyname(h)
            if verbose and attempt > 1:
                print(f"[net-wait] network up after {attempt} attempt(s)",
                      file=sys.stderr)
            return True
        except socket.gaierror as e:
            if verbose:
                print(f"[net-wait] attempt {attempt}: DNS not ready ({e}); "
                      f"retrying in {interval_seconds}s",
                      file=sys.stderr)
            time.sleep(interval_seconds)
    if verbose:
        print(f"[net-wait] gave up after {timeout_seconds}s — proceeding "
              "anyway (HTTP calls will surface real errors)",
              file=sys.stderr)
    return False


if __name__ == "__main__":
    ok = wait_for_network()
    sys.exit(0 if ok else 1)
