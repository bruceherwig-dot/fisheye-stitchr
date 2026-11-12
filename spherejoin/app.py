"""SphereJoin application entry point."""
from __future__ import annotations

import os
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication, QMainWindow

from .panes import ImagePane  # noqa: F401  (kept importable from here)
from .workspace import Workspace


class MainWindow(QMainWindow):
    """Thin shell around the Workspace."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fisheye StitchR")
        self.resize(1700, 940)
        self.workspace = Workspace()
        self.setCentralWidget(self.workspace)
        file_menu = self.menuBar().addMenu("File")
        for text, key, slot in (("Export...", "Ctrl+E", self.workspace.run_export),):
            act = file_menu.addAction(text)
            act.setShortcut(key)
            act.triggered.connect(slot)
        self._hint = ("Wheel: zoom at cursor.  Drag: pan.  Double-click or F: fit.  1: 100%.  "
                      "Arrows nudge the circle center 1 px (Shift 10 px); Alt/Cmd+Up/Down changes the radius.")
        self.statusBar().showMessage(self._hint)
        self.workspace.note.connect(self.show_note)

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
    win.deleteLater()
    app.processEvents()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
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
