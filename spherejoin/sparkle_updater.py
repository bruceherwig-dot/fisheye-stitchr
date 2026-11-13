"""Auto-update for the Mac app, using the Sparkle framework (loaded through PyObjC at run time).

Sparkle draws its own "A new version is available" window and does the download, the in-place install and
the restart. This module only starts it and lets the File menu ask it to check. It does nothing when the
program runs from source (no Sparkle framework present), so development launches are unaffected.

How it reaches users (plain English): the built app carries the web address of a small "feed" file and the
public half of our signing key. Sparkle fetches the feed once a day, and when the user picks Check for
Updates. If the feed lists a newer version, Sparkle shows its install window. See AUTO_UPDATE.md.

Two traps, both learned the hard way in Star Trail CleanR (see its AUTO_UPDATE.md):
1. Keep a module-level reference to the Sparkle controller, or Python frees it mid-check and the app crashes.
2. Never add a delegate method that takes a BOOL argument without an explicit PyObjC signature: PyObjC
   reads the BOOL as a pointer and the app dies with a segfault the moment Sparkle shows an update.
"""
from __future__ import annotations

import os
import sys
import time
import traceback

_updater_controller = None      # strong reference so Python cannot free it
_updater_delegate = None        # same

_LOG_PATH = os.path.expanduser("~/.fisheye_stitchr/sparkle_debug.log")


def _log(msg: str):
    """Timestamped line in the diagnostic log. Best effort; windowed Mac apps have no visible stderr."""
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        with open(_LOG_PATH, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
    except Exception:
        pass


def bring_app_to_front():
    """Pull the app forward so Sparkle's window does not open behind the main window."""
    if sys.platform != "darwin":
        return
    try:
        import objc
        objc.lookUpClass("NSApplication").sharedApplication().activateIgnoringOtherApps_(True)
    except Exception:
        _log(f"bring_app_to_front: EXCEPTION\n{traceback.format_exc()}")


def _bundled_framework_path():
    """Path to Sparkle.framework inside the built .app, or None when running from source."""
    if sys.platform != "darwin" or not hasattr(sys, "_MEIPASS"):
        return None
    candidate = os.path.abspath(os.path.join(os.path.dirname(sys.executable), "..", "Frameworks", "Sparkle.framework"))
    return candidate if os.path.exists(candidate) else None


def init_sparkle() -> bool:
    """Start Sparkle. Call once, after the QApplication exists. Returns True when the engine is running."""
    global _updater_controller, _updater_delegate
    _log("=" * 40)
    path = _bundled_framework_path()
    if path is None:
        _log("init_sparkle: no bundled framework (running from source); nothing to start")
        return False
    try:
        import objc
        ns = {}
        objc.loadBundle("Sparkle", ns, bundle_path=path)
        controller_class = ns.get("SPUStandardUpdaterController")
        if controller_class is None:
            _log("init_sparkle: SPUStandardUpdaterController missing from the framework")
            return False
        NSObject = objc.lookUpClass("NSObject")      # from the bridge already bundled; Foundation is not bundled

        class _Delegate(NSObject):
            def updater_didFindValidUpdate_(self, updater, item):
                """A newer version exists: bring the app forward just before Sparkle's window appears."""
                _log("delegate: update found")
                bring_app_to_front()

            def standardUserDriverWillShowModalAlert(self):
                bring_app_to_front()

        _updater_delegate = _Delegate.alloc().init()
        _updater_controller = controller_class.alloc().initWithStartingUpdater_updaterDelegate_userDriverDelegate_(
            True, _updater_delegate, _updater_delegate)
        _log("init_sparkle: started")
        return True
    except Exception:
        _log(f"init_sparkle: EXCEPTION\n{traceback.format_exc()}")
        return False


def updater_alive() -> bool:
    return _updater_controller is not None


def check_for_updates() -> bool:
    """The user asked to check. True when handed to Sparkle, False when the engine is not running."""
    if _updater_controller is None:
        _log("check_for_updates: engine not running")
        return False
    try:
        _updater_controller.checkForUpdates_(None)
        return True
    except Exception:
        _log(f"check_for_updates: EXCEPTION\n{traceback.format_exc()}")
        return False
