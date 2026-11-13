"""The working area: both source panes, the live preview and the controls, wired together."""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import asdict

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication, QComboBox, QHBoxLayout, QLabel, QMessageBox, QProgressDialog, QPushButton, QScrollArea, QSplitter,
    QStackedWidget, QVBoxLayout, QWidget,
)

from .controls import ControlsPanel
from . import autolevel, dual, exposure
from .imageio_utils import load_image, open_failure_text
from .export import (ExportCancelled, ExportSettings, default_output_path, export_panorama, read_icc,
                     recommended_width)
from .export_dialog import ExportDialog
from .panes import DualDropArea, ImagePane
from .project import (PROJECT_SUFFIX, ProjectError, image_entry, load_project,
                      resolve_image, save_project)
from .preview import (compose_preview, make_proxy, scale_lens, view_for_center, warp_preview_layers)
from . import ui_state
from .reproject import output_size
from .seammask import SeamMask
from .preview_pane import PreviewPane
from .projection import LensParams
from .viewer360 import Viewer360

PREVIEW_W, PREVIEW_H = 2048, 1024
QUICK_W = 1024
SHARP_360_W = 6144     # width of the panorama behind the 360 viewer


class Workspace(QWidget):
    note = Signal(str)     # a plain message for the window's status bar

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pane_a = ImagePane("A (front)")
        self.pane_b = ImagePane("B (back)")
        self.controls = ControlsPanel()
        self.preview = PreviewPane()
        self.viewer360: Viewer360 | None = None
        self.last_preview = None            # latest full-size preview, uint8 RGB
        self._proxy = {"a": None, "b": None}  # (shrunk image, (sx, sy))
        self._layers = None
        self._layers_key = None
        self._layers_width = PREVIEW_W
        self._geo_seen = None
        self._shown_once = False
        self.export_settings = ExportSettings()
        self.seam_mask = SeamMask()
        self._last_paint = None
        self.project_path: str | None = None

        sources = QSplitter(Qt.Horizontal)
        sources.addWidget(self.pane_a)
        sources.addWidget(self.pane_b)
        stack = QSplitter(Qt.Vertical)
        self.dual_drop = DualDropArea()                  # the single drop target of one-photo mode, until a photo is in
        self.source_stack = QStackedWidget()
        self.source_stack.addWidget(self.dual_drop)
        self.source_stack.addWidget(sources)
        stack.addWidget(self.source_stack)
        stack.addWidget(self.preview)
        stack.setSizes([330, 520])
        self.input_box = QComboBox()
        self.input_box.addItem("Two separate photos", "two")
        self.input_box.addItem("One photo with both lenses", "one")
        self.input_box.setToolTip("Two separate photos: drop the front and back photos (together or one at a time). "
                                  "One photo with both lenses: drop a single picture with both fisheye circles "
                                  "side by side, such as an Insta360 .insp, and it is split for you.")
        input_bar = QHBoxLayout()
        input_bar.setContentsMargins(6, 4, 6, 0)
        input_bar.addWidget(QLabel("Input"))
        input_bar.addWidget(self.input_box)
        input_bar.addStretch(1)
        left = QVBoxLayout()
        left.setContentsMargins(0, 0, 0, 0)
        left.addLayout(input_bar)
        left.addWidget(stack, 1)
        self._gain_key = None               # lenses the cached brightness-match gains were measured for
        self._gains = None
        self._lens_saved = None             # lens settings from before a camera preset replaced them
        self.input_mode = ui_state.get_str("input_mode") or "two"
        if self.input_mode not in ("two", "one"):
            self.input_mode = "two"
        self.input_box.setCurrentIndex(0 if self.input_mode == "two" else 1)
        self.pane_a.set_input_mode(self.input_mode)
        self.source_stack.setCurrentIndex(0 if self.input_mode == "one" else 1)
        self.pane_b.set_input_mode(self.input_mode)
        scroll = QScrollArea()
        scroll.setWidget(self.controls)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        self.export_btn = QPushButton("Export...")
        buttons = QHBoxLayout()
        buttons.addWidget(self.export_btn)
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.addLayout(buttons)
        right.addWidget(scroll, 1)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addLayout(left, 1)
        lay.addLayout(right)

        self._quick = self._timer(30, self._do_quick)
        self._full = self._timer(300, self._do_full)
        self._compose = self._timer(30, self._do_compose)
        self._quick_width = QUICK_W

        self.pane_a.files_dropped.connect(lambda paths: self._on_files_dropped("a", paths))
        self.pane_b.files_dropped.connect(lambda paths: self._on_files_dropped("b", paths))
        self.pane_a.image_loaded.connect(lambda: self._on_loaded("a"))
        self.pane_b.image_loaded.connect(lambda: self._on_loaded("b"))
        self.pane_a.circle_changed.connect(self._on_change)
        self.pane_b.circle_changed.connect(self._on_change)
        self.controls.changed.connect(self._on_change)
        self.preview.options_changed.connect(self._on_change)
        self.preview.flicker_tick.connect(self._on_flicker)
        self.preview.open_360.connect(self.open_360)
        self.preview.view.picked.connect(self._on_pick)
        self.preview.view.set_turn_mode(True)
        self.preview.view.view_drag.connect(self._on_view_drag)
        self.preview.view.paint_event.connect(self._on_paint)
        self.preview.undo_requested.connect(self.undo_paint)
        self.preview.clear_requested.connect(self.clear_mask)
        self.export_btn.clicked.connect(self.run_export)
        self.dual_drop.files_dropped.connect(lambda paths: self._on_files_dropped("a", paths))
        self.controls.level_btn.clicked.connect(self.auto_level)
        self.input_box.currentIndexChanged.connect(lambda _=None: self._set_input_mode(self.input_box.currentData()))

    def _timer(self, ms, slot):
        t = QTimer(self)
        t.setSingleShot(True)
        t.setInterval(ms)
        t.timeout.connect(slot)
        return t

    # ---- state -> parameters ----
    def ready(self) -> bool:
        return all(self._proxy[k] is not None for k in "ab") and \
            self.pane_a.circle() is not None and self.pane_b.circle() is not None

    def blend_params(self):
        """The blend settings, plus automatic sky brightness/colour matching between the lenses: the strip of sky both
        lenses see should look the same, so the two lenses' skies are nudged toward each other (the ground, which
        already agrees, is left alone)."""
        bp = self.controls.blend()
        if self.ready():
            lens_a, lens_b = self.lens_pair()
            key = (lens_a, lens_b)
            if self._gain_key != key:
                (pa, sa), (pb, sb) = self._proxy["a"], self._proxy["b"]
                self._gains = exposure.estimate_gains(pa, pb, scale_lens(lens_a, sa), scale_lens(lens_b, sb))
                self._gain_key = key
            bp.sky_gain_a = tuple(float(v) for v in self._gains[0])
            bp.sky_gain_b = tuple(float(v) for v in self._gains[1])
        return bp

    def lens_pair(self):
        """Full-size LensParams for A and B, from the circles and the controls."""
        s = self.controls.lens_settings()
        out = []
        for pane, base, trim in ((self.pane_a, 0.0, s["trim_a"]), (self.pane_b, 180.0, s["trim_b"])):
            cx, cy, r = pane.circle()
            out.append(LensParams(cx=cx, cy=cy, radius=max(r, 1.0), fov_deg=s["fov_deg"], model=s["model"],
                                  poly=s["poly"], trim_yaw=trim[0], trim_pitch=trim[1], trim_roll=trim[2],
                                  base_yaw=base))
        return out[0], out[1]

    def _on_loaded(self, which: str):
        pane = self.pane_a if which == "a" else self.pane_b
        self._proxy[which] = make_proxy(pane.array())
        self.source_stack.setCurrentIndex(1)               # a picture is in: show the pictures, not the drop area
        self._gain_key = None               # a new picture needs new brightness-match measurements
        self._layers_key = None
        self._on_change()

    # ---- scheduling ----
    def _geo_key(self):
        if not self.ready():
            return None
        a, b = self.lens_pair()
        return (a, b, self.controls.view(), self.controls.projection())

    def _on_change(self):
        if not self.ready():
            return
        geo = self._geo_key()
        if geo != self._geo_seen:
            self._geo_seen = geo
            self._quick.start()
            self._full.start()      # restarts while the user is still dragging
        else:
            self._compose.start()

    def _on_flicker(self):
        if self.ready() and self.preview.mode() == "flicker":
            self.render(self._layers_width)

    def _do_quick(self):
        self.render(self._quick_width)

    def _do_full(self):
        self._quick.stop()
        self.render(PREVIEW_W)

    def _do_compose(self):
        self.render(self._layers_width)

    def flush(self):
        """Stop pending timers and render the full-size preview now."""
        for t in (self._quick, self._full, self._compose):
            t.stop()
        if self.ready():
            self._geo_seen = self._geo_key()
            self.render(PREVIEW_W)

    # ---- rendering ----
    def render(self, width: int):
        if not self.ready():
            return None
        t0 = time.perf_counter()
        lens_a, lens_b = self.lens_pair()
        view = self.controls.view()
        proj = self.controls.projection()
        (pa, sa), (pb, sb) = self._proxy["a"], self._proxy["b"]
        la, lb = scale_lens(lens_a, sa), scale_lens(lens_b, sb)
        w, h = output_size(proj, width)
        key = (width, la, lb, view, proj)
        if key != self._layers_key:
            self._layers = warp_preview_layers(pa, pb, la, lb, view, w, h, proj=proj)
            self._layers_key = key
            self._layers_width = width
        bp = self.blend_params()
        out = compose_preview(self._layers, la, lb, bp, mode=self.preview.mode(),
                              show_seam=self.preview.chk_seam.isChecked(),
                              show_overlap=self.preview.chk_overlap.isChecked(),
                              flicker_phase=self.preview.flicker_phase, mask=self.seam_mask,
                              show_mask=self.preview.chk_mask.isChecked())
        # The view always works in full-size preview units, so a quick small render stands in for it.
        self.preview.view.set_image(out, display_size=output_size(proj, PREVIEW_W), keep_view=self._shown_once)
        self._shown_once = True
        ms = (time.perf_counter() - t0) * 1000.0
        self.preview.status.setText(f"preview {w} x {h}   {ms:.0f} ms"
                                    f"{'  (quick, refining...)' if width < PREVIEW_W else ''}")
        if width == PREVIEW_W:
            self.last_preview = out
            if self.viewer360 is not None and self.viewer360.isVisible():
                self.viewer360.set_panorama(self._sphere_for_360(out))
        return out

    def _sphere_for_360(self, shown):
        """The full-sphere (equirectangular) picture the 360 viewer needs. When the preview is already
        equirectangular that is the preview itself; otherwise render it separately."""
        proj = self.controls.projection()
        if proj.is_equirect:
            return shown
        lens_a, lens_b = self.lens_pair()
        (pa, sa), (pb, sb) = self._proxy["a"], self._proxy["b"]
        la, lb = scale_lens(lens_a, sa), scale_lens(lens_b, sb)
        layers = warp_preview_layers(pa, pb, la, lb, self.controls.view(), PREVIEW_W, PREVIEW_W // 2)
        return compose_preview(layers, la, lb, self.blend_params(), mode="blend")

    # ---- interactions ----
    def _on_pick(self, sx: float, sy: float):
        proj = self.controls.projection()
        pw, ph = output_size(proj, PREVIEW_W)
        v = view_for_center(self.controls.view(), sx, sy, pw, ph, proj)
        self.preview.pick_btn.setChecked(False)
        self.controls.set_view(v.yaw, v.pitch)

    def _on_view_drag(self, sx: float, sy: float, phase: int):
        """Dragging the preview turns the output view (yaw only; pitch and roll are left alone)."""
        proj = self.controls.projection()
        pw, ph = output_size(proj, PREVIEW_W)
        if phase == 0:
            self._turn_start = (self.controls.view(), sx, sy)
            return
        if getattr(self, "_turn_start", None) is None:
            return
        v0, sx0, sy0 = self._turn_start
        # The spot that started at the press has to follow the cursor, so the point that many pixels the
        # other way becomes the center of the view.
        if proj.is_planet:
            yaw = view_for_center(v0, pw / 2.0 - (sx - sx0), sy0, pw, ph, proj).yaw
        else:
            yaw = v0.yaw - (sx - sx0) * 360.0 / pw            # the picture is 360 degrees wide
        yaw = (yaw + 180.0) % 360.0 - 180.0
        self.controls.set_view(yaw, v0.pitch)
        if phase == 2:
            self._turn_start = None

    def open_360(self):
        if self.viewer360 is None:
            self.viewer360 = Viewer360()
        if self.last_preview is None and self.ready():
            self.flush()
        if self.last_preview is not None:
            self.viewer360.set_panorama(self._sphere_for_360(self.last_preview))
        self.viewer360.show()
        self.viewer360.raise_()
        if self.ready():
            self._sharpen_360()

    def _sharpen_360(self):
        """Replace the quick small panorama with one built from the full-size photos, so the viewer is sharp
        when zoomed in. Takes a few seconds; the quick one stays on screen until it is done."""
        from .blend import compose
        from .reproject import EQUIRECT
        self.note.emit("Sharpening the 360 view from your full-size photos...")
        QApplication.processEvents()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            lens_a, lens_b = self.lens_pair()
            pano = compose(self.pane_a.array(), self.pane_b.array(), lens_a, lens_b, self.controls.view(),
                           self.blend_params(), SHARP_360_W, SHARP_360_W // 2, interp="linear",
                           proj=EQUIRECT, mask=self.seam_mask)
            self.viewer360.set_panorama(pano)
            self.note.emit("360 view is now full quality.")
        finally:
            QApplication.restoreOverrideCursor()

    # ---- seam mask painting ----
    def _on_paint(self, x: float, y: float, phase: int):
        """One mouse event of a brush stroke on the preview (phase 0 press, 1 move, 2 release)."""
        if not self.ready():
            return
        proj = self.controls.projection()
        pw, ph = output_size(proj, PREVIEW_W)
        view = self.controls.view()
        r = self.preview.brush_radius()
        tool = self.preview.paint_tool()
        if phase == 0:
            self.seam_mask.push_undo()          # one undo step per stroke
            self._last_paint = None
        pts = [(x, y)] if self._last_paint is None else [self._last_paint, (x, y)]
        self.seam_mask.stroke(view, proj, pw, ph, pts, r, tool)
        self._last_paint = None if phase == 2 else (x, y)
        if not self._compose.isActive():       # do not restart it on every move, or a long drag never redraws
            self._compose.start()

    def undo_paint(self):
        if self.seam_mask.undo() and self.ready():
            self._compose.start()

    def clear_mask(self):
        if not self.seam_mask.is_empty:
            self.seam_mask.clear()
            if self.ready():
                self._compose.start()

    def _set_input_mode(self, mode: str):
        """Switch between two separate photos and one two-lens photo. Both panes start empty so the two halves of
        one photo are never mistaken for two photos."""
        if mode == self.input_mode:
            return
        self.input_mode = mode
        self.source_stack.setCurrentIndex(0 if mode == "one" else 1)      # one drop area, or the two pictures
        ui_state.set_str("input_mode", mode)
        if mode == "two":
            self._restore_lens()
        for pane in (self.pane_a, self.pane_b):
            pane.clear()
            pane.set_input_mode(mode)
        self._proxy = {"a": None, "b": None}
        self._layers = None
        self._layers_key = None
        self._geo_seen = None
        self.last_preview = None
        self.preview.view.set_image(None)
        self._shown_once = False
        self.note.emit("Drop your photo with both lenses on either picture." if mode == "one"
                       else "Drop your front and back photos on A and B.")

    def auto_level(self):
        """Line the two lenses up from the star trails, level the pair from the horizon and face the pole."""
        if not self.ready():
            self.note.emit("Load both pictures and set both circles first.")
            return
        lens_a, lens_b = self.lens_pair()
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = autolevel.auto_level(self.pane_a.array(), lens_a, self.pane_b.array(), lens_b)
        finally:
            QApplication.restoreOverrideCursor()
        if res.trim_a is None:
            self.note.emit(" ".join(res.notes))
            return
        self.controls.apply_lens_preset(res.fov_deg, res.model, res.trim_a, res.trim_b)
        if res.view_yaw is not None:
            self.controls.set_view(res.view_yaw, 0.0, 0.0)
        self.note.emit(f"Lined up the two skies ({res.model} lens, field of view {res.fov_deg:.1f}), leveled and "
                       "faced the pole. " + " ".join(res.notes))

    def _restore_lens(self):
        """Hand back the lens settings a camera preset replaced."""
        if self._lens_saved is not None:
            fov, model, trim_a, trim_b = self._lens_saved
            self._lens_saved = None
            self.controls.apply_lens_preset(fov, model, trim_a, trim_b)

    def load_dual(self, path: str) -> bool:
        """Fill A and B from one photo that holds both lens circles side by side."""
        try:
            arr = load_image(path)
        except Exception as e:
            QMessageBox.warning(self, "Could not open image", open_failure_text(path, e))
            return False
        left, right = dual.split_halves(arr)
        preset = dual.preset_for(path, (arr.shape[1], arr.shape[0]))
        name = os.path.basename(path)
        self.pane_a.load_array(left, path, f"{name} (left half)", dual.circle_for_half(left, preset, "a"))
        self.pane_b.load_array(right, path, f"{name} (right half)", dual.circle_for_half(right, preset, "b"))
        ui_state.set_str("last_files/both", os.path.abspath(path))
        self.source_stack.setCurrentIndex(1)                                # now show the two halves
        if preset is None:
            self._restore_lens()
        if preset is not None:
            if self._lens_saved is None:
                s = self.controls.lens_settings()
                self._lens_saved = (s["fov_deg"], s["model"], tuple(s["trim_a"]), tuple(s["trim_b"]))
            self.controls.apply_lens_preset(preset.fov_deg, preset.model, preset.trim_a, preset.trim_b)
            self.note.emit(f"Recognized an {preset.name} photo: circles and lens settings are set.")
        elif not dual.looks_two_to_one(arr):
            self.note.emit("That photo is not twice as wide as it is tall, so it may not hold two lens circles. "
                           "Check the circles.")
        else:
            self.note.emit("Split the photo down the middle. Check the circles and the field of view.")
        return True

    def _on_files_dropped(self, which: str, paths: list):
        """One picture goes onto the pane it was dropped on. Two or more fill A then B, whichever pane got them."""
        if self.input_mode == "one":
            self.load_dual(paths[0])
            if len(paths) > 1:
                self.note.emit(f"Used {os.path.basename(paths[0])}; this mode takes one photo with both lenses.")
            return
        if len(paths) >= 2:
            self.pane_a.load(paths[0])
            self.pane_b.load(paths[1])
            note = " (used the first two)" if len(paths) > 2 else ""
            self.preview.status.setText(f"loaded {os.path.basename(paths[0])} as A and {os.path.basename(paths[1])} as B{note}")
        else:
            (self.pane_a if which == "a" else self.pane_b).load(paths[0])

    def open_last_files(self) -> list:
        """Reopen the two photos from last time. A photo that has moved or whose drive is not connected is
        skipped quietly. Returns the paths that were opened."""
        opened = []
        if self.input_mode == "one":
            p = ui_state.get_str("last_files/both")
            if p and os.path.isfile(p) and self.load_dual(p):
                self.note.emit("Reopened your last photo. Drop a new photo on the A or B picture to replace it.")
                return [p]
            return []
        for pane, key in ((self.pane_a, "last_files/a"), (self.pane_b, "last_files/b")):
            p = ui_state.get_str(key)
            if p and os.path.isfile(p):
                pane.load(p)
                opened.append(p)
        if opened:
            self.note.emit("Reopened your last photos. Drop new photos on the A or B picture to replace them.")
        return opened

    def close_children(self):
        if self.viewer360 is not None:
            self.viewer360.close()

    # ---- project save / load ----
    def project_dict(self, project_path: str) -> dict:
        d = self.controls.state_dict()
        d["images"] = {"a": image_entry(self.pane_a.path, project_path), "b": image_entry(self.pane_b.path, project_path)}
        d["circles"] = {"a": list(self.pane_a.circle() or []) or None, "b": list(self.pane_b.circle() or []) or None}
        d["preview"] = {"mode": self.preview.mode(), "seam_line": self.preview.chk_seam.isChecked(),
                        "overlap_zone": self.preview.chk_overlap.isChecked()}
        d["export"] = asdict(self.export_settings)
        if not self.seam_mask.is_empty:
            d["seam_mask"] = os.path.basename(self._mask_path(project_path))
        return d

    @staticmethod
    def _mask_path(project_path: str) -> str:
        base = project_path[:-len(PROJECT_SUFFIX)] if project_path.endswith(PROJECT_SUFFIX) else os.path.splitext(project_path)[0]
        return base + ".seammask.png"

    def save_project_to(self, path: str):
        save_project(path, self.project_dict(path))
        if not self.seam_mask.is_empty:
            self.seam_mask.save_png(self._mask_path(path))
        self.project_path = path

    def load_project_from(self, path: str, interactive: bool = False) -> list:
        """Load a project file. Returns a list of warnings (also shown when interactive)."""
        warnings: list[str] = []
        if self.input_mode != "two":                       # a project holds two separate pictures
            self.input_box.setCurrentIndex(0)
        try:
            d = load_project(path)
        except ProjectError as e:
            if interactive:
                QMessageBox.warning(self, "Could not open project", str(e))
            return [str(e)]
        for key, pane in (("a", self.pane_a), ("b", self.pane_b)):
            entry = (d.get("images") or {}).get(key)
            found = resolve_image(entry, path)
            if found is None:
                if entry:
                    warnings.append(f"Image {key.upper()} was not found: {entry.get('path')}")
                continue
            pane.load(found)
            c = (d.get("circles") or {}).get(key)
            if c and len(c) == 3:
                pane.set_circle(float(c[0]), float(c[1]), float(c[2]))
        self.controls.load_state(d)
        self.seam_mask = SeamMask()
        name = d.get("seam_mask")
        if name:
            mp = os.path.join(os.path.dirname(os.path.abspath(path)), name)
            try:
                self.seam_mask = SeamMask.load_png(mp)
            except (OSError, ValueError):
                warnings.append(f"The painted seam mask file was not found: {mp}")
        self._layers_key = None               # layers cache the mask look-up, so rebuild them
        pv = d.get("preview") or {}
        from .preview import MODES
        if pv.get("mode") in MODES:
            self.preview.mode_box.setCurrentIndex(MODES.index(pv["mode"]))
        self.preview.chk_seam.setChecked(bool(pv.get("seam_line", False)))
        self.preview.chk_overlap.setChecked(bool(pv.get("overlap_zone", False)))
        ex = d.get("export") or {}
        try:
            self.export_settings = ExportSettings(**{k: v for k, v in ex.items() if k in ExportSettings.__dataclass_fields__})
        except TypeError:
            self.export_settings = ExportSettings()
        self.project_path = path
        if interactive and warnings:
            QMessageBox.warning(self, "Project opened with problems", "\n".join(warnings))
        return warnings

    # ---- export ----
    def run_export(self):
        if not self.ready():
            QMessageBox.information(self, "Export", "Load both images and set both circles first.")
            return
        lens_a, lens_b = self.lens_pair()
        proj = self.controls.projection()
        rec = recommended_width(lens_a, lens_b)
        if proj.is_planet:
            rec = max(1024, round(rec / 2 / 16) * 16)      # a planet is square and shows less sky per pixel row
        st = self.export_settings
        if not st.width or st.width == ExportSettings().width:
            st.width = rec
        out0 = default_output_path(self.pane_a.path, st.fmt, proj.kind)
        dlg = ExportDialog(self, st, out0, rec, self.pane_a.array().dtype.itemsize == 2, proj=proj,
                           has_mask=not self.seam_mask.is_empty)
        if dlg.exec() != ExportDialog.Accepted:
            return
        st = dlg.settings()
        self.export_settings = st
        prog = QProgressDialog("Exporting...", "Cancel", 0, 100, self)
        prog.setWindowTitle("Export")
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(0)
        prog.setValue(0)

        def cb(frac, text):
            prog.setLabelText(text)
            prog.setValue(int(frac * 100))
            QApplication.processEvents()
            if prog.wasCanceled():
                raise ExportCancelled()

        try:
            result = export_panorama(dlg.path(), self.pane_a.array(), self.pane_b.array(), lens_a, lens_b,
                                     self.controls.view(), self.blend_params(), st,
                                     icc=read_icc(self.pane_a.path), protect=(self.pane_a.path, self.pane_b.path),
                                     progress=cb, proj=proj, mask=self.seam_mask)
        except ExportCancelled:
            prog.close()
            self.preview.status.setText("export cancelled; nothing was written for the cancelled step")
            return
        except Exception as e:
            prog.close()
            QMessageBox.warning(self, "Export failed", str(e))
            return
        prog.close()
        # No confirmation box: the file opens selected in Finder, and the result goes in the status line.
        note = "  |  " + " ".join(result.messages) if result.messages else ""
        self.preview.status.setText(f"exported {os.path.basename(result.paths[0])}"
                                    f"{' and ' + str(len(result.paths) - 1) + ' layers' if len(result.paths) > 1 else ''}{note}")
        try:
            subprocess.run(["open", "-R", result.paths[0]], check=False)  # show it in Finder
        except OSError:
            pass
