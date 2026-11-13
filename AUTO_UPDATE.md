# How Fisheye StitchR auto-update works (read this first)

Verified against the code on 2026-11-12. Do not describe update behavior from memory; check the files cited.

## The short version
- The Mac app carries Sparkle (an update engine) inside it, a web address for a small "feed" file, and the
  PUBLIC half of our signing key. All set by `build_app.py` (Info.plist keys SUFeedURL, SUPublicEDKey,
  SUEnableAutomaticChecks, SUScheduledCheckInterval; CFBundleVersion is the plain number, e.g. 0.10).
- The feed is `appcast-mac-apple-silicon.xml` on GitHub Pages (branch `gh-pages` of this repo):
  https://bruceherwig-dot.github.io/fisheye-stitchr/appcast-mac-apple-silicon.xml
- Sparkle checks the feed once a day, and when the user picks File > Check for Updates... If the top entry is
  newer than the running app, Sparkle shows its install window, downloads the DMG named in the entry (a GitHub
  release asset), verifies its signature against the public key, installs in place and restarts.
- The app code is `spherejoin/sparkle_updater.py`; it starts at launch from `spherejoin/app.py`. It does
  nothing when running from source (no Sparkle framework present).

## Publishing (automatic, on every `v*` tag)
`.github/workflows/build.yml`: build-mac (tests, build, launch check, UPDATER-ALIVE GATE, DMG) -> release ->
publish-appcast. publish-appcast signs the DMG with the `SPARKLE_ED_PRIVATE_KEY` secret (via `scripts/publish_appcast.py`
and the vendored `sign_update`), adds the entry to the feed on `gh-pages`, pushes, then FAILS unless the live feed
advertises the new version. "Release done" = publish-appcast green, not just a green build.

## Keys
- Private key: GitHub secret `SPARKLE_ED_PRIVATE_KEY`, plus a backup on Bruce's Mac at
  `~/.fisheye_stitchr/sparkle_private_key.txt` (chmod 600). Never commit it. If it is lost, installed apps can
  never be updated again (they only trust this key), so keep the backup.
- Public key: `assets/sparkle_public_key.txt` (baked into every build).
- This key is separate from Star Trail CleanR's.

## Version numbers
Tag `v0.10-beta` -> Sparkle compares the number `0.10` part by part (0.10 is newer than 0.9). Window title and
About show `0.10-beta` (from `version.txt`).

## Bootstrapping
v0.9-beta has no updater. v0.10-beta is the first version that updates itself and must be installed by hand once.
An updater fix only reaches users who already have an updater-capable build.

## Two traps carried over from Star Trail CleanR
- Keep the Sparkle controller referenced at module level (else Python frees it mid-check and the app crashes).
- Never add a delegate method with a BOOL argument without an explicit PyObjC signature (segfault when Sparkle
  shows an update).
- macOS silently disables Sparkle when the app runs from the DMG or from a quarantined download location. The
  DMG shows an Applications shortcut so people drag it across.

## Proof that counts: a real frozen build
CI and tests cannot see the Python / Sparkle / macOS boundary. Before calling an updater change done, install a real
build, publish a throwaway higher version (`vX.YY-test`) so the feed has something to offer, open the installed app,
File > Check for Updates..., confirm Sparkle's window appears with no crash, click Skip. Then delete the throwaway
release and tag and remove its entry from the feed. Diagnostic log on every machine:
`~/.fisheye_stitchr/sparkle_debug.log`.
