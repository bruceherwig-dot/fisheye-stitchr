"""Project files: all settings in one JSON file saved next to the images."""
from __future__ import annotations

import json
import os

PROJECT_VERSION = 1
PROJECT_SUFFIX = ".spherejoin.json"


class ProjectError(Exception):
    pass


def default_project_path(path_a: str | None) -> str:
    if not path_a:
        return os.path.join(os.path.expanduser("~"), "untitled" + PROJECT_SUFFIX)
    stem = os.path.splitext(os.path.basename(path_a))[0]
    return os.path.join(os.path.dirname(path_a), stem + PROJECT_SUFFIX)


def image_entry(path: str | None, project_path: str) -> dict | None:
    if not path:
        return None
    entry = {"path": os.path.abspath(path)}
    try:
        entry["relative"] = os.path.relpath(os.path.abspath(path), os.path.dirname(os.path.abspath(project_path)))
    except ValueError:
        pass
    return entry


def resolve_image(entry: dict | None, project_path: str) -> str | None:
    """Prefer the path relative to the project file (the folder may have moved), then the absolute one."""
    if not entry:
        return None
    base = os.path.dirname(os.path.abspath(project_path))
    if entry.get("relative"):
        p = os.path.normpath(os.path.join(base, entry["relative"]))
        if os.path.isfile(p):
            return p
    p = entry.get("path")
    return p if p and os.path.isfile(p) else None


def save_project(path: str, data: dict):
    data = dict(data)
    data["app"] = "SphereJoin"
    data["version"] = PROJECT_VERSION
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, path)  # a crash mid-save never leaves a half-written project


def load_project(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ProjectError(f"Project file not found: {path}")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ProjectError(f"This is not a readable project file ({e}).")
    if not isinstance(data, dict) or data.get("app") != "SphereJoin":
        raise ProjectError("This file is not a SphereJoin project.")
    if int(data.get("version", 0)) > PROJECT_VERSION:
        raise ProjectError("This project was saved by a newer version of SphereJoin.")
    return data
