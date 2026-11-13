import json
import os

import pytest

from spherejoin.project import (
    ProjectError, default_project_path, image_entry, load_project, resolve_image, save_project,
)


def test_save_load_roundtrip_and_atomic_write(tmp_path):
    p = tmp_path / "a.spherejoin.json"
    save_project(str(p), {"circles": {"a": [1, 2, 3]}})
    d = load_project(str(p))
    assert d["circles"]["a"] == [1, 2, 3] and d["app"] == "SphereJoin" and d["version"] == 1
    assert not (tmp_path / "a.spherejoin.json.tmp").exists()


def test_bad_files_give_plain_errors(tmp_path):
    with pytest.raises(ProjectError):
        load_project(str(tmp_path / "missing.json"))
    junk = tmp_path / "junk.json"
    junk.write_text("{not json")
    with pytest.raises(ProjectError):
        load_project(str(junk))
    other = tmp_path / "other.json"
    other.write_text(json.dumps({"app": "SomethingElse"}))
    with pytest.raises(ProjectError):
        load_project(str(other))
    newer = tmp_path / "newer.json"
    newer.write_text(json.dumps({"app": "SphereJoin", "version": 99}))
    with pytest.raises(ProjectError):
        load_project(str(newer))


def test_image_found_by_relative_path_after_the_folder_moves(tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    img = old / "a.jpg"
    img.write_bytes(b"x")
    proj = old / "p.spherejoin.json"
    entry = image_entry(str(img), str(proj))
    assert resolve_image(entry, str(proj)) == str(img)
    new = tmp_path / "new"
    old.rename(new)
    assert resolve_image(entry, str(new / "p.spherejoin.json")) == str(new / "a.jpg")
    (new / "a.jpg").unlink()
    assert resolve_image(entry, str(new / "p.spherejoin.json")) is None
    assert resolve_image(None, str(proj)) is None


def test_default_path_is_next_to_image_a():
    assert os.path.normpath(default_project_path("/x/y/a.jpg")) == os.path.normpath("/x/y/a.spherejoin.json")
