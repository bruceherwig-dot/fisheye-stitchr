"""Build gate (Windows): the built .exe must carry our update-signing public key as the resource WinSparkle reads.

WinSparkle looks for a resource of type "EdDSA" named "EdDSAPub" in the running .exe (Windows treats resource names
as case-insensitive and stores them in capitals). If it is missing, the engine cannot check update signatures at all.
Usage: python scripts/check_win_resource.py path\\to\\app.exe
"""
import pathlib
import sys

import pefile

exe = sys.argv[1]
want = (pathlib.Path(__file__).resolve().parent.parent / "assets" / "sparkle_public_key.txt").read_text().strip()
pe = pefile.PE(exe)


def text(name):
    return name.decode() if isinstance(name, bytes) else str(name)


found = None
listing = []
root = getattr(pe, "DIRECTORY_ENTRY_RESOURCE", None)
for kind in (root.entries if root else []):
    kind_name = text(kind.name) if kind.name is not None else f"#{kind.id}"
    for item in kind.directory.entries:
        item_name = text(item.name) if item.name is not None else f"#{item.id}"
        listing.append(f"{kind_name}/{item_name}")
        if kind_name.upper() == "EDDSA" and item_name.upper() == "EDDSAPUB":
            leaf = item.directory.entries[0].data.struct
            found = pe.get_data(leaf.OffsetToData, leaf.Size).decode("ascii", "replace").strip("\x00 \r\n")
print("resources in the .exe:", ", ".join(listing) or "(none)")
if found is None:
    sys.exit("FAILED: the .exe has no EdDSA / EdDSAPub resource, so update signatures could not be checked.")
if found != want:
    sys.exit(f"FAILED: the resource in the .exe ({found!r}) is not our public key.")
print("OK: the .exe carries our update-signing public key.")
