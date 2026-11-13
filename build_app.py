"""Build Fisheye StitchR with PyInstaller: the Mac app (dist/Fisheye StitchR.app) or the Windows folder
(dist/FisheyeStitchR/FisheyeStitchR.exe). Neither is code-signed or notarized. The auto-updater is inside.

Run:  .venv/bin/python build_app.py        (on a Mac, or on Windows: python build_app.py)
"""
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent
VERSION = (ROOT / "version.txt").read_text().strip()
PLIST_VERSION = VERSION.split("-")[0]       # the bundle wants plain numbers; "-beta" shows in the window title
WIN = sys.platform == "win32"
NAME = "FisheyeStitchR" if WIN else "Fisheye StitchR"
# Where installed Mac apps look for updates (a small file on GitHub Pages, written by CI on every release).
FEED_URL = "https://bruceherwig-dot.github.io/fisheye-stitchr/appcast-mac-apple-silicon.xml"

# Qt pieces this app never uses; leaving them out keeps the download small.
EXCLUDES = ["PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
            "PySide6.Qt3DCore", "PySide6.QtMultimedia", "PySide6.QtPdf", "PySide6.QtCharts",
            "PySide6.QtDataVisualization", "PySide6.QtSql", "PySide6.QtTest", "tkinter", "pytest"]

# PyInstaller refuses to start from inside the package folder; use a tiny entry script at the top level.
ENTRY = ROOT / "build" / "entry.py"


def run_pyinstaller():
    (ROOT / "build").mkdir(exist_ok=True)
    ENTRY.write_text("from spherejoin.app import main\nmain()\n")
    for d in ("dist", "build/" + NAME):
        shutil.rmtree(ROOT / d, ignore_errors=True)
    pubkey = (ROOT / "assets" / "sparkle_public_key.txt").read_text().strip()
    cmd = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", NAME,
           "--paths", str(ROOT), "--distpath", str(ROOT / "dist"), "--workpath", str(ROOT / "build"),
           "--specpath", str(ROOT / "build"), "--hidden-import", "spherejoin.app",
           "--add-data", f"{ROOT / 'version.txt'}{os.pathsep}."]
    if WIN:
        # WinSparkle checks update signatures against a public key stored as a resource named EdDSAPub
        # (type EdDSA) inside the .exe. Without it, it would accept an unsigned update.
        key_file = ROOT / "build" / "eddsa_pub.bin"
        key_file.write_bytes(pubkey.encode("ascii"))
        cmd += ["--icon", str(ROOT / "assets" / "icon.ico"), "--resource", f"{key_file},EdDSA,EdDSAPub"]
    else:
        cmd += ["--icon", str(ROOT / "assets" / "icon.icns"), "--hidden-import", "objc",
                "--osx-bundle-identifier", "com.bruceherwig.fisheyestitchr"]
    for m in EXCLUDES:
        cmd += ["--exclude-module", m]
    cmd.append(str(ENTRY))
    subprocess.run(cmd, check=True, cwd=ROOT)


def finish_mac():
    app = ROOT / "dist" / f"{NAME}.app"
    plist = app / "Contents" / "Info.plist"

    # 1. Sparkle.framework goes inside the app. ditto, not cp -R: cp breaks the framework's symlinks.
    sparkle_dest = app / "Contents" / "Frameworks" / "Sparkle.framework"
    sparkle_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(sparkle_dest, ignore_errors=True)
    subprocess.run(["ditto", str(ROOT / "vendored" / "Sparkle.framework"), str(sparkle_dest)], check=True)
    # A loose file at the framework's top level makes the code-signing check fail ("unsealed contents").
    # Sparkle's license still ships with the app, in the app's Resources folder.
    lic = sparkle_dest / "LICENSE.txt"
    if lic.exists():
        shutil.move(str(lic), str(app / "Contents" / "Resources" / "Sparkle-LICENSE.txt"))

    # 2. Version and Sparkle settings in Info.plist. Sparkle refuses to start without CFBundleVersion.
    pubkey = (ROOT / "assets" / "sparkle_public_key.txt").read_text().strip()
    keys = [("CFBundleShortVersionString", "string", PLIST_VERSION), ("CFBundleVersion", "string", PLIST_VERSION),
            ("SUFeedURL", "string", FEED_URL), ("SUPublicEDKey", "string", pubkey),
            ("SUEnableAutomaticChecks", "bool", "true"), ("SUScheduledCheckInterval", "integer", "86400")]
    for key, kind, value in keys:
        ok = subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Set :{key} {value}", str(plist)],
                            capture_output=True).returncode == 0
        if not ok:
            subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Add :{key} {kind} {value}", str(plist)], check=True)

    # 3. Re-seal the outer app with an ad-hoc signature. Adding the framework and editing Info.plist broke
    # PyInstaller's seal, and Sparkle rejects updates from an app whose outer seal is broken.
    subprocess.run(["codesign", "--force", "--sign", "-", str(app)], check=True)
    subprocess.run(["codesign", "--verify", "--verbose=2", str(app)], check=True)
    print("Built", app)


def finish_windows():
    out = ROOT / "dist" / NAME
    # WinSparkle.dll goes next to the .exe so Windows' normal DLL search finds it.
    shutil.copy2(ROOT / "vendored" / "winsparkle" / "WinSparkle.dll", out / "WinSparkle.dll")
    shutil.copy2(ROOT / "vendored" / "winsparkle" / "COPYING.txt", out / "WinSparkle-LICENSE.txt")
    print("Built", out)


def main():
    run_pyinstaller()
    finish_windows() if WIN else finish_mac()


if __name__ == "__main__":
    main()
