"""SphereJoin application entry point."""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox

from . import __version__, sparkle_updater, winsparkle_updater
from .panes import ImagePane  # noqa: F401  (kept importable from here)
from .workspace import Workspace


# Where installed Windows apps look for updates (written by CI on every release; see AUTO_UPDATE.md).
WIN_FEED_URL = "https://bruceherwig-dot.github.io/fisheye-stitchr/appcast-windows.xml"
RELEASES_URL = "https://github.com/bruceherwig-dot/fisheye-stitchr/releases/latest"


def numeric_version() -> str:
    """'0.10-beta' -> '0.10': the plain number the update engines compare."""
    return __version__.split("-")[0]


class MainWindow(QMainWindow):
    """Thin shell around the Workspace."""

    # Outcomes of a quiet Windows update check. WinSparkle calls back on its own thread; emitting a signal
    # hops to the UI thread, where dialogs are allowed.
    update_found = Signal()
    update_current = Signal()
    update_failed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Fisheye StitchR {__version__}")
        self.resize(1700, 940)
        self.workspace = Workspace()
        self.setCentralWidget(self.workspace)
        file_menu = self.menuBar().addMenu("File")
        for text, key, slot in (("Export...", "Ctrl+E", self.workspace.run_export),):
            act = file_menu.addAction(text)
            act.setShortcut(key)
            act.triggered.connect(slot)
        upd = file_menu.addAction("Check for Updates...")
        upd.triggered.connect(self.check_for_updates)
        about = file_menu.addAction("About Fisheye StitchR")
        about.setMenuRole(QAction.AboutRole)
        about.triggered.connect(self.show_about)
        self._hint = ("Wheel: zoom at cursor.  Drag: pan.  Double-click or F: fit.  1: 100%.  "
                      "Arrows nudge the circle center 1 px (Shift 10 px); Alt/Cmd+Up/Down changes the radius.")
        self.statusBar().showMessage(self._hint)
        self.workspace.note.connect(self.show_note)
        self.update_found.connect(winsparkle_updater.check_for_updates)       # confirmed: show the engine's window
        self.update_current.connect(lambda: QMessageBox.information(
            self, "Check for Updates", f"You are up to date. Fisheye StitchR {__version__} is the newest version."))
        self.update_failed.connect(self._update_check_failed)
        winsparkle_updater.set_handlers(self.update_found.emit, self.update_current.emit, self.update_failed.emit)

    def _update_check_failed(self):
        box = QMessageBox(self)
        box.setWindowTitle("Check for Updates")
        box.setText("Could not check for updates.")
        box.setInformativeText("Security software or a network setting may be blocking the check. "
                               "You can download the newest version from GitHub.")
        get = box.addButton("Open the download page", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Close)
        box.exec()
        if box.clickedButton() is get:
            QDesktopServices.openUrl(QUrl(RELEASES_URL))

    def check_for_updates(self):
        if sys.platform == "win32":
            # Quiet first: nothing from the engine's own windows unless it confirms a newer version exists.
            if winsparkle_updater.check_for_updates_quiet() or winsparkle_updater.check_for_updates():
                return
        elif sparkle_updater.check_for_updates():
            return
        QMessageBox.information(
            self, "Check for Updates",
            "Updates are checked in the installed app. Download the newest version from the "
            "Fisheye StitchR page on GitHub.")

    def show_about(self):
        QMessageBox.about(
            self, "About Fisheye StitchR",
            f"<b>Fisheye StitchR {__version__}</b><br>Joins two back-to-back fisheye pictures into one 360-style panorama.")

    def show_note(self, text: str, ms: int = 12000):
        self.statusBar().showMessage(text, ms)
        QTimer.singleShot(ms + 100, lambda: self.statusBar().showMessage(self._hint))

    def closeEvent(self, e):
        self.workspace.close_children()
        super().closeEvent(e)


def _shutdown(app: QApplication, win: MainWindow, code: int):
    """Quit without Python's normal interpreter shutdown.

    On exit PySide tears the Qt objects down in an order that segfaults here (a button deleted after
    its parent), which put a "Python quit unexpectedly" dialog on screen every time the app closed.
    Everything is already saved or written by the time we get here, so close the child windows,
    flush the output streams and leave directly.
    """
    win.workspace.close_children()
    winsparkle_updater.cleanup()               # stop the Windows update engine's thread before the process ends
    win.deleteLater()
    app.processEvents()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


def _live_update_check(app: QApplication, win: MainWindow):
    """Build gate (Windows): do a REAL update check against the live feed and exit with the result.
    "The engine loads" was never proof that a check works. Exit 0: feed fetched and parsed (update found or
    already current). 1: the engine could not start a check. 2: the engine reported an error. 3: no answer in 90 s."""
    import time
    result = {"v": None}
    winsparkle_updater.set_handlers(lambda: result.__setitem__("v", "found"),
                                    lambda: result.__setitem__("v", "current"),
                                    lambda: result.__setitem__("v", "error"))
    if not winsparkle_updater.check_for_updates_quiet():
        winsparkle_updater.cleanup()
        os._exit(1)
    end = time.time() + 90
    while result["v"] is None and time.time() < end:
        app.processEvents()
        time.sleep(0.1)
    code = {"found": 0, "current": 0, "error": 2}.get(result["v"], 3)
    winsparkle_updater.cleanup()
    os._exit(code)


def run(argv: list[str]):
    """Start the app. A hidden --quit-after-ms=N flag closes it after N milliseconds (for testing)."""
    app = QApplication(argv)
    win = MainWindow()
    win.show()
    quit_ms = None
    project = None
    paths = []
    for a in argv[1:]:
        if a.startswith("--quit-after-ms="):
            quit_ms = int(a.split("=", 1)[1])
        elif a.startswith("--project="):
            project = a.split("=", 1)[1]           # developer shortcut: load saved settings, no menu for it
        elif not a.startswith("-"):
            paths.append(a)
    if sys.platform == "win32":
        winsparkle_updater.init_winsparkle(WIN_FEED_URL, "Fisheye StitchR", numeric_version(), "Fisheye StitchR")
        alive = winsparkle_updater.updater_alive
    else:
        sparkle_updater.init_sparkle()
        alive = sparkle_updater.updater_alive
    if os.environ.get("FISHEYE_UPDATER_SMOKE") == "1":
        # Build gate: the built app must prove its update engine came alive, then leave. (A Windows GUI program
        # has no console, so the answer is the exit code.)
        ok = alive()
        winsparkle_updater.cleanup()
        os._exit(0 if ok else 1)
    if os.environ.get("FISHEYE_UPDATER_CHECK") == "1" and sys.platform == "win32":
        _live_update_check(app, win)
    ws = win.workspace
    if ws.input_mode == "one" and paths:
        ws.load_dual(paths[0])
    else:
        if len(paths) > 0:
            ws.pane_a.load(paths[0])
        if len(paths) > 1:
            ws.pane_b.load(paths[1])
    if project:
        ws.load_project_from(project)
    elif not paths:
        opened = ws.open_last_files()          # nothing was named, so pick up where the last session ended
        if opened:
            print("reopened last files:", *opened, flush=True)
    if quit_ms is not None:
        QTimer.singleShot(quit_ms, app.quit)
    _shutdown(app, win, app.exec())


def main():
    run(sys.argv)
