#!/usr/bin/env python3
"""Publish a release to the auto-update feed (the "appcast"), and check that it is live.

Used by the publish-appcast job in .github/workflows/build.yml on every version tag.

  publish:  python3 scripts/publish_appcast.py v0.10-beta --artifacts-dir DIR --key-file KEY --pages-dir DIR
            For each platform (Mac, Windows): signs the installer found under the artifacts folder with Sparkle's
            sign_update and puts a new entry at the top of that platform's feed file inside the pages folder
            (a checkout of the gh-pages branch). It does not commit or push.
  verify:   python3 scripts/publish_appcast.py v0.10-beta --verify
            Polls every live feed until it advertises that version; exits nonzero if one never does.

Older entries in the feed are never touched (the new entry is spliced in as text).
"""
import argparse
import glob
import os
import re
import subprocess
import sys
import time
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SIGN_UPDATE = REPO_ROOT / "vendored" / "sparkle-bin" / "sign_update"
SPARKLE_NS = "http://www.andymatuschak.org/xml-namespaces/sparkle"
PAGES_BASE = "https://bruceherwig-dot.github.io/fisheye-stitchr"
RELEASE_BASE = "https://github.com/bruceherwig-dot/fisheye-stitchr/releases/download"
MIN_MACOS = "11.0"

# One entry per platform: the installer's file name in the release, and the feed file that advertises it.
PLATFORMS = [
    {"key": "mac-apple-silicon", "release_filename": "FisheyeStitchR-Mac-AppleSilicon.dmg",
     "appcast": "appcast-mac-apple-silicon.xml",
     "extra": f"            <sparkle:minimumSystemVersion>{MIN_MACOS}</sparkle:minimumSystemVersion>\n"},
    # THE WINDOWS ENCLOSURE MUST BE THE .exe, NOT THE .zip. WinSparkle hands the downloaded file to Windows
    # with no arguments and lets the file association decide, so a zip merely opens Explorer and installs
    # NOTHING (Star Trail CleanR shipped that bug for months). The .zip still ships for people downloading by
    # hand (it dodges Edge SmartScreen). Do not "tidy" these back together.
    {"key": "windows", "release_filename": "FisheyeStitchRSetup.exe", "appcast": "appcast-windows.xml", "extra": ""},
]
LIVE_FEEDS = {p["key"]: f"{PAGES_BASE}/{p['appcast']}" for p in PLATFORMS}
LIVE_FEED = LIVE_FEEDS["mac-apple-silicon"]          # kept for the tests that compare it with build_app.py


def parse_version(tag: str):
    """'v0.10-beta' -> ('0.10', '0.10-beta'): the number Sparkle compares, and the label users see."""
    m = re.match(r"v(\d+(?:\.\d+)*)", tag)
    if not m:
        sys.exit(f"Could not read a version from tag '{tag}'. Expected something like v0.10-beta.")
    return m.group(1), tag.lstrip("v")


def top_item_version(xml_text: str):
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    item = root.find(".//item")
    return None if item is None else item.findtext(f"{{{SPARKLE_NS}}}version")


def sign_file(path: Path, key_file: Path):
    """Returns (ed signature, size in bytes) for the installer."""
    r = subprocess.run([str(SIGN_UPDATE), "-p", "-f", str(key_file), str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"sign_update failed:\n{r.stderr}")
    return r.stdout.strip(), os.path.getsize(path)


def build_item(platform, tag, numeric, short, sig, length, pub_date):
    fn = platform["release_filename"]
    if platform["key"] == "windows" and not fn.lower().endswith((".exe", ".msi")):
        sys.exit(f"Refusing to publish a Windows feed that points at {fn}: Windows cannot install that by "
                 "executing it, so every one-click update would silently do nothing.")
    url = f"{RELEASE_BASE}/{tag}/{fn}"
    return (
        "        <item>\n"
        f"            <title>Version {short}</title>\n"
        f"            <pubDate>{pub_date}</pubDate>\n"
        f"            <sparkle:version>{numeric}</sparkle:version>\n"
        f"            <sparkle:shortVersionString>{short}</sparkle:shortVersionString>\n"
        f"{platform['extra']}"
        f'            <enclosure url="{url}" length="{length}" type="application/octet-stream" '
        f'sparkle:edSignature="{sig}"/>\n'
        "        </item>\n"
    )


def insert_item(xml_text: str, item_xml: str) -> str:
    m = re.search(r"<language>[^<]*</language>\s*\n", xml_text)
    if m:
        at = m.end()
    else:
        m2 = re.search(r"[ \t]*<item>", xml_text)
        if not m2:
            sys.exit("The feed has no <language> line and no <item> to anchor against.")
        at = m2.start()
    return xml_text[:at] + item_xml + xml_text[at:]


def find_installer(artifacts_dir, filename):
    hits = glob.glob(os.path.join(artifacts_dir, "**", filename), recursive=True)
    if not hits:
        sys.exit(f"Installer '{filename}' not found anywhere under {artifacts_dir}")
    return Path(hits[0])


def do_publish(args):
    numeric, short = parse_version(args.tag)
    if not SIGN_UPDATE.exists():
        sys.exit(f"Missing signing tool: {SIGN_UPDATE}")
    if not Path(args.key_file).exists():
        sys.exit(f"Signing key not found: {args.key_file}")
    pub_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    for platform in PLATFORMS:
        print(f"=== {platform['key']} ===")
        xml_path = Path(args.pages_dir) / platform["appcast"]
        if not xml_path.exists():
            sys.exit(f"Feed file not found: {xml_path}")
        text = xml_path.read_text()
        if top_item_version(text) == numeric:
            print(f"  already advertises {numeric} at the top; leaving it alone.")
            continue
        installer = find_installer(args.artifacts_dir, platform["release_filename"])
        sig, length = sign_file(installer, Path(args.key_file))
        item = build_item(platform, args.tag, numeric, short, sig, length, pub_date)
        xml_path.write_text(insert_item(text, item))
        print(f"  added Version {short} ({length / 1048576:.1f} MB) to {xml_path.name}")


def fetch(url: str, timeout=20) -> str:
    req = urllib.request.Request(f"{url}?nocache={int(time.time())}", headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def do_verify(args):
    numeric, _ = parse_version(args.tag)
    deadline = time.time() + args.timeout
    pending = dict(LIVE_FEEDS)
    last = {}
    while pending and time.time() < deadline:
        for key, url in list(pending.items()):
            try:
                last[key] = top_item_version(fetch(url))
            except Exception as e:      # the site can take a few minutes to update after a push
                last[key] = f"(fetch failed: {e})"
            if last[key] == numeric:
                print(f"{key}: live feed advertises {numeric}.")
                del pending[key]
        if pending:
            print("waiting for:", ", ".join(f"{k} (shows {last.get(k)})" for k in pending))
            time.sleep(15)
    if pending:
        sys.exit(f"FAILED: these live feeds never advertised {numeric}: {', '.join(pending)}. "
                 "Auto-update users on those platforms did NOT get this release.")
    print("All live feeds advertise the new version. Auto-update users will get this release.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--artifacts-dir")
    ap.add_argument("--key-file")
    ap.add_argument("--pages-dir")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="seconds to wait in verify mode")
    args = ap.parse_args()
    if args.verify:
        do_verify(args)
    else:
        if not (args.artifacts_dir and args.key_file and args.pages_dir):
            sys.exit("publish mode needs --artifacts-dir, --key-file and --pages-dir")
        do_publish(args)


if __name__ == "__main__":
    main()
