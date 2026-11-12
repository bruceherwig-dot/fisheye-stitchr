"""Remembers small interface choices (which sections are open) between sessions."""
from __future__ import annotations

import os

from PySide6.QtCore import QSettings


def settings() -> QSettings:
    """The user's saved interface state. Tests point SPHEREJOIN_SETTINGS_FILE at a scratch file so they
    never touch the real one."""
    path = os.environ.get("SPHEREJOIN_SETTINGS_FILE")
    if path:
        return QSettings(path, QSettings.IniFormat)
    return QSettings("SphereJoin", "SphereJoin")


def get_bool(key: str, default: bool) -> bool:
    v = settings().value(key, None)
    if v is None:
        return default
    return str(v).lower() in ("true", "1", "yes")


def set_bool(key: str, value: bool):
    s = settings()
    s.setValue(key, bool(value))
    s.sync()


def get_str(key: str, default: str = "") -> str:
    v = settings().value(key, None)
    return default if v is None else str(v)


def set_str(key: str, value: str):
    s = settings()
    s.setValue(key, str(value))
    s.sync()
