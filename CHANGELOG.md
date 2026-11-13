# Fisheye StitchR, Version History

---

## v0.10-beta (2026-11-12)
First version that updates itself.
- Automatic updates through Sparkle: the app checks once a day and from File > Check for Updates..., shows its own "new version" window, then downloads, installs and restarts. Updates are signed; the app refuses anything not signed with our key.
- Opening the app never shows an error for a photo it could not reopen (a drive that is off, or a folder macOS blocks). The photos are skipped quietly.
- When a photo you open cannot be read, the message now says why (not found, no permission, unsupported type, damaged file).
- Version number shown in the window title and in About. New app icon.
- Note: 0.9-beta has no updater, so install 0.10-beta by hand once; later versions update themselves.

## v0.9-beta (2026-11-12)
First version, before any public release. Stitches two back-to-back fisheye pictures, or one Insta360 X4 photo with both lenses, into a 360-style panorama.
- Two input modes: two separate photos, or one photo with both lenses (Insta360 X4 recognized by its camera tag or its exact picture size).
- Layouts: equirectangular, Mercator, Gall and tiny planet, with a Planet/Tunnel slider.
- Automatic brightness and colour matching between the lenses, an equidistant seam blend, a painted seam, and a full-resolution 360 viewer.
- "Find the pole and level" for star-trail pairs.
- Drag the preview to turn the view, double-click to zoom. Export to JPEG (default), TIFF or PNG.
