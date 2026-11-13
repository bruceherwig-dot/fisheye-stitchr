"""Auto-update for the Windows app, using WinSparkle (WinSparkle.dll loaded with ctypes).

WinSparkle draws its own "update available" window, downloads the installer and runs it. This module only
starts it and lets the File menu ask it to check. It does nothing when running from source or on other systems.

Lessons carried over from Star Trail CleanR (see its AUTO_UPDATE.md):
- win_sparkle_set_appcast_url takes a NARROW string (const char *). Declaring it wide makes the engine read
  the URL as the single letter "h", and every check fails while the error blames the user's connection.
- WinSparkle runs the downloaded file with no arguments, so the feed must advertise the .exe installer,
  never a .zip (a zip just opens in Explorer and installs nothing).
- A user-initiated check runs quietly first (no engine windows). Only a confirmed find opens the engine's
  install window; "up to date" and "error" are shown by our own dialogs, so a machine where security software
  blocks the engine's networking still gets a plain explanation and a download link.
"""
from __future__ import annotations

import ctypes
import os
import sys

_dll = None
_LOG_PATH = os.path.expanduser("~/.fisheye_stitchr/winsparkle_debug.log")


def _log(msg: str):
    """Timestamped line in a diagnostic log (a windowed Windows app has no console). Best effort."""
    try:
        import time
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


_CB = ctypes.CFUNCTYPE(None)        # void (*)(void), the shape of every callback below

# Kept at module level so ctypes does not free them while WinSparkle holds their pointers.
_error_cb = _found_cb = _notfound_cb = None
_error_handler = _found_handler = _notfound_handler = None
_user_initiated = False             # a quiet background failure must never pop a dialog


def _fire(handler):
    """Run a handler the GUI registered (a Qt signal's emit, so it hops to the UI thread). Never raises."""
    global _user_initiated
    try:
        if _user_initiated and handler is not None:
            handler()
    except Exception:
        pass
    _user_initiated = False


def _on_error():
    _fire(_error_handler)


def _on_found():
    _fire(_found_handler)


def _on_notfound():
    _fire(_notfound_handler)


def set_handlers(found, notfound, error):
    """Register thread-safe callables for the three outcomes of a quiet check."""
    global _found_handler, _notfound_handler, _error_handler
    _found_handler, _notfound_handler, _error_handler = found, notfound, error


def _find_dll():
    """WinSparkle.dll next to the .exe inside the built app, or None (source run / not Windows)."""
    if sys.platform != "win32" or not hasattr(sys, "_MEIPASS"):
        return None
    for c in (os.path.join(os.path.dirname(sys.executable), "WinSparkle.dll"),
              os.path.join(sys._MEIPASS, "WinSparkle.dll")):
        if os.path.exists(c):
            return c
    return None


def init_winsparkle(appcast_url: str, app_name: str, app_version: str, company_name: str) -> bool:
    """Start WinSparkle. Call once at startup. Returns True when the engine is running."""
    global _dll, _error_cb, _found_cb, _notfound_cb
    path = _find_dll()
    _log(f"init_winsparkle: dll path = {path}")
    if path is None:
        return False
    try:
        dll = ctypes.CDLL(path)
        _log("init_winsparkle: dll loaded")
        dll.win_sparkle_set_appcast_url.argtypes = [ctypes.c_char_p]            # NARROW, see the note above
        dll.win_sparkle_set_app_details.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_wchar_p]
        dll.win_sparkle_set_automatic_check_for_updates.argtypes = [ctypes.c_int]
        dll.win_sparkle_init.argtypes = []
        dll.win_sparkle_check_update_with_ui.argtypes = []
        dll.win_sparkle_check_update_without_ui.argtypes = []
        dll.win_sparkle_cleanup.argtypes = []
        # All configuration must come before win_sparkle_init().
        dll.win_sparkle_set_appcast_url(appcast_url.encode("ascii"))
        dll.win_sparkle_set_app_details(company_name, app_name, app_version)
        dll.win_sparkle_set_automatic_check_for_updates(1)
        try:        # older DLLs may lack these exports; then quiet checks are simply unavailable
            dll.win_sparkle_set_error_callback.argtypes = [_CB]
            _error_cb = _CB(_on_error)
            dll.win_sparkle_set_error_callback(_error_cb)
            dll.win_sparkle_set_did_find_update_callback.argtypes = [_CB]
            _found_cb = _CB(_on_found)
            dll.win_sparkle_set_did_find_update_callback(_found_cb)
            dll.win_sparkle_set_did_not_find_update_callback.argtypes = [_CB]
            _notfound_cb = _CB(_on_notfound)
            dll.win_sparkle_set_did_not_find_update_callback(_notfound_cb)
        except Exception:
            _found_cb = _notfound_cb = None
        _log("init_winsparkle: configured; calling win_sparkle_init")
        dll.win_sparkle_init()
        _dll = dll
        _log("init_winsparkle: started")
        return True
    except Exception as e:
        _log(f"init_winsparkle: EXCEPTION {e!r}")
        _dll = None
        return False


def updater_alive() -> bool:
    return _dll is not None


def check_for_updates_quiet() -> bool:
    """User-initiated check with no engine windows. The result arrives through the registered handlers.
    False when the engine is not running or this DLL lacks the outcome callbacks (caller falls back)."""
    global _user_initiated
    if _dll is None or _found_cb is None:
        return False
    try:
        _user_initiated = True
        _dll.win_sparkle_check_update_without_ui()
        return True
    except Exception:
        return False


def check_for_updates() -> bool:
    """Foreground check with the engine's own windows (used after a quiet check confirmed an update)."""
    global _user_initiated
    if _dll is None:
        return False
    try:
        _user_initiated = True
        _dll.win_sparkle_check_update_with_ui()
        return True
    except Exception:
        return False


# NOTE: there is deliberately no cleanup() here. Calling win_sparkle_cleanup() before exit crashed the built app on
# Windows (access violation, CI run of 2026-11-12), and Star Trail CleanR never calls it either.
