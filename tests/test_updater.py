"""Guards for the auto-update path. These are structural checks: the real proof is a frozen build doing a
full update (see AUTO_UPDATE.md)."""
import base64
import importlib.util
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _publisher():
    spec = importlib.util.spec_from_file_location("publish_appcast", ROOT / "scripts" / "publish_appcast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FEED = ('<rss><channel><language>en</language>\n'
        '<item><sparkle:version xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">0.9</sparkle:version></item>\n'
        '</channel></rss>')


def test_version_is_read_from_the_tag():
    p = _publisher()
    assert p.parse_version("v0.10-beta") == ("0.10", "0.10-beta")
    assert p.parse_version("v1.0") == ("1.0", "1.0")


def test_new_entry_goes_on_top_and_old_ones_are_untouched():
    p = _publisher()
    item = p.build_item("v0.10-beta", "0.10", "0.10-beta", "SIG", 123, "Fri, 13 Nov 2026 00:00:00 +0000")
    out = p.insert_item(FEED, item)
    assert out.index("0.10-beta") < out.index("<item><sparkle:version")
    assert FEED.split("\n")[1] in out                                   # the old entry is byte-for-byte there
    assert 'sparkle:edSignature="SIG"' in out and "releases/download/v0.10-beta/" in out


def test_the_public_key_is_a_real_32_byte_key():
    key = (ROOT / "assets" / "sparkle_public_key.txt").read_text().strip()
    assert len(base64.b64decode(key)) == 32


def test_the_build_wires_sparkle_into_the_app():
    build = (ROOT / "build_app.py").read_text()
    for needle in ("SUFeedURL", "SUPublicEDKey", "SUEnableAutomaticChecks", "Sparkle.framework", "codesign"):
        assert needle in build
    assert (ROOT / "vendored" / "Sparkle.framework" / "Versions").is_dir()
    assert (ROOT / "vendored" / "sparkle-bin" / "sign_update").exists()


def test_the_feed_address_in_the_app_matches_the_publisher():
    build = (ROOT / "build_app.py").read_text()
    m = re.search(r'FEED_URL = "([^"]+)"', build)
    assert m and m.group(1) == _publisher().LIVE_FEED


def test_ci_gates_and_publishes():
    wf = (ROOT / ".github" / "workflows" / "build.yml").read_text()
    assert "FISHEYE_UPDATER_SMOKE=1" in wf          # the built app must prove its update engine started
    assert "publish-appcast:" in wf and "--verify" in wf


def test_running_from_source_does_not_start_an_updater():
    from spherejoin import sparkle_updater
    assert sparkle_updater.init_sparkle() is False
    assert sparkle_updater.check_for_updates() is False
