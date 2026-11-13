#!/usr/bin/env python3
"""Publish a release to the auto-update feed (the "appcast"), and check that it is live.

Used by the publish-appcast job in .github/workflows/build.yml on every version tag.

  publish:  python3 scripts/publish_appcast.py v0.10-beta --installer FILE.dmg --key-file KEY --pages-dir DIR
            Signs the installer with Sparkle's sign_update and puts a new entry at the top of the feed
            file inside DIR (a checkout of the gh-pages branch). It does not commit or push.
  verify:   python3 scripts/publish_appcast.py v0.10-beta --verify
            Polls the live feed until it advertises that version; exits nonzero if it never does.

Older entries in the feed are never touched (the new entry is spliced in as text).
"""
import argparse
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
APPCAST_NAME = "appcast-mac-apple-silicon.xml"
LIVE_FEED = "https://bruceherwig-dot.github.io/fisheye-stitchr/" + APPCAST_NAME
RELEASE_BASE = "https://github.com/bruceherwig-dot/fisheye-stitchr/releases/download"
INSTALLER_NAME = "FisheyeStitchR-Mac-AppleSilicon.dmg"
MIN_MACOS = "11.0"


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


def build_item(tag, numeric, short, sig, length, pub_date):
    url = f"{RELEASE_BASE}/{tag}/{INSTALLER_NAME}"
    return (
        "        <item>\n"
        f"            <title>Version {short}</title>\n"
        f"            <pubDate>{pub_date}</pubDate>\n"
        f"            <sparkle:version>{numeric}</sparkle:version>\n"
        f"            <sparkle:shortVersionString>{short}</sparkle:shortVersionString>\n"
        f"            <sparkle:minimumSystemVersion>{MIN_MACOS}</sparkle:minimumSystemVersion>\n"
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


def do_publish(args):
    numeric, short = parse_version(args.tag)
    if not SIGN_UPDATE.exists():
        sys.exit(f"Missing signing tool: {SIGN_UPDATE}")
    if not Path(args.key_file).exists():
        sys.exit(f"Signing key not found: {args.key_file}")
    xml_path = Path(args.pages_dir) / APPCAST_NAME
    if not xml_path.exists():
        sys.exit(f"Feed file not found: {xml_path}")
    text = xml_path.read_text()
    if top_item_version(text) == numeric:
        print(f"Feed already advertises {numeric} at the top; leaving it alone.")
        return
    sig, length = sign_file(Path(args.installer), Path(args.key_file))
    pub_date = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")
    xml_path.write_text(insert_item(text, build_item(args.tag, numeric, short, sig, length, pub_date)))
    print(f"Added Version {short} ({length / 1048576:.1f} MB) to {xml_path.name}.")


def fetch(url: str, timeout=20) -> str:
    req = urllib.request.Request(f"{url}?nocache={int(time.time())}", headers={"Cache-Control": "no-cache"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def do_verify(args):
    numeric, _ = parse_version(args.tag)
    deadline = time.time() + args.timeout
    seen = None
    while time.time() < deadline:
        try:
            seen = top_item_version(fetch(LIVE_FEED))
        except Exception as e:      # the site can take a few minutes to update after a push
            seen = f"(fetch failed: {e})"
        if seen == numeric:
            print(f"Live feed advertises {numeric}. Auto-update users will get this release.")
            return
        print(f"Live feed shows {seen}, waiting for {numeric} ...")
        time.sleep(15)
    sys.exit(f"FAILED: the live feed never advertised {numeric} (last saw {seen}). "
             "Auto-update users did NOT get this release.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--installer")
    ap.add_argument("--key-file")
    ap.add_argument("--pages-dir")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--timeout", type=int, default=600, help="seconds to wait in verify mode")
    args = ap.parse_args()
    if args.verify:
        do_verify(args)
    else:
        if not (args.installer and args.key_file and args.pages_dir):
            sys.exit("publish mode needs --installer, --key-file and --pages-dir")
        do_publish(args)


if __name__ == "__main__":
    main()
