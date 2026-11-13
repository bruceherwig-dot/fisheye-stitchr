"""Build the Fisheye StitchR Mac app (unsigned) with PyInstaller.

Run:  .venv/bin/python build_app.py
Output: dist/Fisheye StitchR.app
"""
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = (ROOT / "version.txt").read_text().strip()
NAME = "Fisheye StitchR"

# Qt pieces this app never uses; leaving them out keeps the download small.
EXCLUDES = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
            "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
            "PySide6.QtDataVisualization", "PySide6.QtSql", "PySide6.QtTest", "tkinter", "pytest"]

# PyInstaller refuses to start from inside the package folder; use a tiny entry script at the top level.
ENTRY = ROOT / "build" / "entry.py"


def main():
    (ROOT / "build").mkdir(exist_ok=True)
    ENTRY.write_text("from spherejoin.app import main\nmain()\n")
    for d in ("dist", "build/" + NAME):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed",
           "--name", NAME, "--icon", str(ROOT / "assets" / "icon.icns"),
           "--osx-bundle-identifier", "com.bruceherwig.fisheyestitchr",
           "--paths", str(ROOT), "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"),
           "--specpath", str(ROOT / "build"), "--hidden-import", "spherejoin.app"]
    for m in EXCLUDES:
        cmd += ["--exclude-module", m]
    cmd.append(str(ENTRY))
    subprocess.run(cmd, check=True, cwd=ROOT)
    plist = ROOT / "dist" / f"{NAME}.app" / "Contents" / "Info.plist"
    for key in ("CFBundleShortVersionString", "CFBundleVersion"):
        subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Set :{key} {VERSION}", str(plist)], check=False)
        subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Add :{key} string {VERSION}", str(plist)],
                       check=False, stderr=subprocess.DEVNULL)
    print("Built", ROOT / "dist" / f"{NAME}.app")


if __name__ == "__main__":
    main()
