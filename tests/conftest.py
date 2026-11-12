import gc
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
# Tests keep their own saved interface state so they never touch the real one (or each other).
os.environ["SPHEREJOIN_SETTINGS_FILE"] = os.path.join(tempfile.mkdtemp(prefix="sj_settings_"), "ui.ini")

import pytest


@pytest.fixture(scope="session")
def app():
    """One QApplication for the whole test session."""
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _delete_all_windows():
    from PySide6.QtCore import QCoreApplication, QEvent
    from PySide6.QtWidgets import QApplication
    qapp = QApplication.instance()
    if qapp is None:
        return
    for w in list(QApplication.topLevelWidgets()):
        try:
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    qapp.processEvents()
    gc.collect()


@pytest.fixture(autouse=True)
def _fresh_ui_state(monkeypatch, tmp_path):
    """Every test starts with no remembered open/closed sections."""
    monkeypatch.setenv("SPHEREJOIN_SETTINGS_FILE", str(tmp_path / "ui_state.ini"))


@pytest.fixture(autouse=True)
def _tidy_qt_windows():
    """After every test, delete any windows it left behind so nothing is torn down out of order at exit."""
    yield
    _delete_all_windows()


def pytest_sessionfinish(session, exitstatus):
    _delete_all_windows()
