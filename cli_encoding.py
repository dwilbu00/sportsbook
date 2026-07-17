"""ASCII-safe stdout/stderr for the command-line tools.

The CLI scripts print Unicode (em-dashes, arrows, check marks, mu/sigma/lambda,
box-drawing rules, etc.) in their reports. On Windows the default console
encoding is cp1252/cp437, so those prints raise ``UnicodeEncodeError`` and
crash the tool mid-run. Reconfiguring the standard streams to UTF-8 (with a
non-fatal error handler as a backstop) matches the ``PYTHONUTF8=1`` /
``PYTHONIOENCODING=utf-8`` workaround, so the tools no longer need those
environment variables just to run on Windows.

Call :func:`configure_stdio` once at the top of each CLI ``__main__`` entry
point. It is a harmless no-op on streams that don't support reconfiguration
(already-detached or redirected streams) and on platforms that don't need it.
"""

import sys


def configure_stdio():
    """Best-effort switch of stdout/stderr to UTF-8 so non-ASCII prints don't
    crash on legacy Windows code pages. Safe to call more than once."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="backslashreplace")
        except (ValueError, OSError):
            # Stream doesn't support reconfiguration (detached/redirected).
            continue
