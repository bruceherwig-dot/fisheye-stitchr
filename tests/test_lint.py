"""No undefined names anywhere in the app or its scripts.

Star Trail CleanR shipped a launch crash (name 'screen' is not defined) from a one-line drop in a refactor;
nothing but a fresh user opening the app would have found it. pyflakes finds this class of bug in a second.
Unused imports and the like are tolerated; an undefined name never is. The same check runs in CI before any
build starts. If this fails, fix the name. Do not exclude the file."""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_no_undefined_names():
    r = subprocess.run([sys.executable, "-m", "pyflakes", "spherejoin", "scripts", "build_app.py"],
                       cwd=ROOT, capture_output=True, text=True)
    bad = [ln for ln in (r.stdout + r.stderr).splitlines() if "undefined name" in ln]
    assert not bad, "undefined names found:\n" + "\n".join(bad)
