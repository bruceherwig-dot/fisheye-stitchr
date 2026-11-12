"""Right-hand controls: lens settings, per-lens trims, output view, blend."""
from __future__ import annotations

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QPushButton,
    QSlider, QVBoxLayout, QWidget,
)

from .blend import BlendParams
from .collapsible import CollapsibleSection
from .projection import MODELS, ViewParams
from .reproject import DEFAULT_KIND, EQUIRECT, KIND_LABELS, KINDS, MAX_LAT, PLANET_REACH, Projection


def _spin(lo, hi, step, decimals, value, tip="", width=84):
    sp = QDoubleSpinBox()
    sp.setRange(lo, hi)
    sp.setDecimals(decimals)
    sp.setSingleStep(step)
    sp.setValue(value)
    sp.setKeyboardTracking(False)
    sp.setMinimumWidth(width)
    if tip:
        sp.setToolTip(tip)
    return sp


class _ResetSlider(QSlider):
    """A slider that announces a double-click, which the panel uses to reset it."""
    doubleClicked = Signal()

    def mouseDoubleClickEvent(self, e):
        self.doubleClicked.emit()
        e.accept()


class SliderSpin(QWidget):
    """A slider and a number box that always agree. Double-click the slider to reset it."""
    valueChanged = Signal(float)

    def __init__(self, lo, hi, step, decimals=1, value=0.0):
        super().__init__()
        self.default = value
        self.slider = _ResetSlider(Qt.Horizontal)
        self.slider.setToolTip(f"Double-click to reset to {value:g}")
        self.slider.doubleClicked.connect(lambda: self.setValue(self.default))
        self._k = 10  # slider ticks per unit
        self.slider.setRange(int(lo * self._k), int(hi * self._k))
        self.spin = _spin(lo, hi, step, decimals, value)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.setValue(int(round(value * self._k)))
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)

    def _from_slider(self, v):
        self.spin.blockSignals(True)
        self.spin.setValue(v / self._k)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(v * self._k)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float):
        self.spin.setValue(v)  # fires valueChanged through the spin box


class ControlsPanel(QWidget):
    changed = Signal()

    def __init__(self):
        super().__init__()
        self._sections: list[CollapsibleSection] = []
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)

        # ---- lens ----
        g = QWidget()
        f = QFormLayout(g)
        self.fov = _spin(100, 360, 0.5, 1, 220.0, "Full field of view across the circle, degrees")
        self.model = QComboBox()
        self.model.addItems(MODELS)
        self.k1 = _spin(-5, 5, 0.01, 3, 1.0, "Polynomial: coefficient of u")
        self.k2 = _spin(-5, 5, 0.01, 3, 0.0, "Polynomial: coefficient of u^3")
        self.k3 = _spin(-5, 5, 0.01, 3, 0.0, "Polynomial: coefficient of u^5")
        f.addRow("Field of view", self.fov)
        f.addRow("Model", self.model)
        f.addRow("Poly k1 (u)", self.k1)
        f.addRow("Poly k2 (u^3)", self.k2)
        f.addRow("Poly k3 (u^5)", self.k3)
        outer.addWidget(self._section("Lens settings", g, False, "controls/lens"))
        self.poly_widgets = [self.k1, self.k2, self.k3]

        # ---- trims ----
        self.trim_a = self._trim_group(outer, "Lens A trim (degrees)")
        self.trim_b = self._trim_group(outer, "Lens B trim (degrees)", b=True)

        # ---- output view ----
        g = QWidget()
        f = QFormLayout(g)
        self.v_yaw = SliderSpin(-180, 180, 0.1, 1, 0.0)
        self.v_pitch = SliderSpin(-90, 90, 0.1, 1, 0.0)
        self.v_roll = SliderSpin(-180, 180, 0.1, 1, 0.0)
        f.addRow("Yaw", self.v_yaw)
        f.addRow("Pitch", self.v_pitch)
        f.addRow("Roll", self.v_roll)
        self.reset_view_btn = QPushButton("Reset view")
        f.addRow(self.reset_view_btn)
        self.level_btn = QPushButton("Find the pole and level")
        self.level_btn.setToolTip("For star-trail pictures: finds the horizon and the star-trail pole in each picture, "
                                  "levels both lenses, turns lens B to match lens A, and turns the output to face "
                                  "the pole that is below the horizon so the sky forms a symmetric arch.")
        f.addRow(self.level_btn)
        outer.addWidget(self._section("Output view (what lands in the center)", g, False, "controls/view"))

        g = QWidget()
        f = QFormLayout(g)
        self.proj_kind = QComboBox()
        for k in KINDS:
            self.proj_kind.addItem(KIND_LABELS[k], k)
        self.proj_kind.setToolTip("Equirectangular is the full 360 x 180 sphere. Mercator and Gall keep the "
                                  "star-trail rings around the pole rounder but cut off the top and bottom.")
        self.proj_kind.setCurrentIndex(KINDS.index(DEFAULT_KIND))
        start = Projection.default(DEFAULT_KIND)
        self.lat_top = _spin(5, MAX_LAT, 1.0, 1, start.lat_top_deg, "Highest latitude shown (degrees above the horizon)")
        self.lat_bot = _spin(-MAX_LAT, -5, 1.0, 1, start.lat_bot_deg, "Lowest latitude shown (degrees below the horizon)")
        f.addRow("Layout", self.proj_kind)
        f.addRow("Top latitude", self.lat_top)
        f.addRow("Bottom latitude", self.lat_bot)
        self.planet_tilt = SliderSpin(-90, 90, 0.5, 1, -90.0)
        self.planet_tilt.setToolTip("Left: planet (ground in the middle). Right: tunnel (sky in the middle). "
                                    "This is the Output view pitch, so the two always agree.")
        f.addRow("Planet / Tunnel", self.planet_tilt)
        self._layout_form = f
        self._view_before_planet = None     # (yaw, pitch, roll) to hand back when leaving the tiny planet
        outer.addWidget(self._section("Output layout", g, True, "controls/layout"))

        # ---- blend ----
        g = QWidget()
        f = QFormLayout(g)
        self.seam = _spin(60, 120, 0.5, 1, 90.0, "Angle off lens A's axis where the two sides meet")
        self.feather = _spin(0, 60, 0.5, 1, 20.0, "Width of the soft blend on the ground, centered on the seam")
        self.sky_feather = _spin(0, 60, 0.5, 1, 7.0, "Width of the blend in the sky. Star trails from the two photos "
                                 "do not match, so a narrow blend ghosts them less.")
        self.hard = QCheckBox("Hard cut (no blend)")
        f.addRow("Seam angle", self.seam)
        f.addRow("Ground blend width", self.feather)
        f.addRow("Sky blend width", self.sky_feather)
        f.addRow(self.hard)
        outer.addWidget(self._section("Blend", g, True, "controls/blend"))
        outer.addStretch(1)

        # ---- wiring ----
        every = [self.fov, self.k1, self.k2, self.k3, self.seam, self.feather, self.sky_feather] + list(self.trim_a) + list(self.trim_b)
        for sp in every:
            sp.valueChanged.connect(self._emit)
        for ss in (self.v_yaw, self.v_pitch, self.v_roll):
            ss.valueChanged.connect(self._emit)
        self.v_pitch.valueChanged.connect(self._pitch_to_tilt)
        self.planet_tilt.valueChanged.connect(self._tilt_to_pitch)
        self.model.currentTextChanged.connect(self._model_changed)
        self.hard.toggled.connect(self._emit)
        self.reset_view_btn.clicked.connect(self.reset_view)
        self.lat_top.valueChanged.connect(self._emit)
        self.lat_bot.valueChanged.connect(self._emit)
        self.proj_kind.currentIndexChanged.connect(self._kind_changed)
        self._sync_lat_enabled()
        self._model_changed(self.model.currentText())
        self._install_resets()

    def _section(self, title, content, expanded, key):
        sec = CollapsibleSection(title, content, expanded, key)
        self._sections.append(sec)
        return sec

    def expand_all(self):
        for sec in self._sections:
            sec.setExpanded(True)

    def collapse_all(self):
        for sec in self._sections:
            sec.setExpanded(False)

    # ---- double-click a label to reset that control ----
    def _install_resets(self):
        """Double-clicking a row's label puts that control back to its default."""
        def to(sp, v):
            return lambda: sp.setValue(v)

        r = {
            self.fov: (to(self.fov, 220.0), "220"),
            self.model: (lambda: self.model.setCurrentIndex(0), MODELS[0]),
            self.k1: (to(self.k1, 1.0), "1"), self.k2: (to(self.k2, 0.0), "0"), self.k3: (to(self.k3, 0.0), "0"),
            self.proj_kind: (lambda: self.proj_kind.setCurrentIndex(KINDS.index(DEFAULT_KIND)), DEFAULT_KIND),
            self.lat_top: (lambda: self._reset_lat(self.lat_top, "lat_top_deg"), "the layout's usual value"),
            self.lat_bot: (lambda: self._reset_lat(self.lat_bot, "lat_bot_deg"), "the layout's usual value"),
            self.v_yaw: (to(self.v_yaw, 0.0), "0"), self.v_pitch: (to(self.v_pitch, 0.0), "0"),
            self.v_roll: (to(self.v_roll, 0.0), "0"),
            self.planet_tilt: (to(self.planet_tilt, -90.0), "-90 (planet)"),
            self.seam: (to(self.seam, 90.0), "90"), self.feather: (to(self.feather, 20.0), "20"),
            self.sky_feather: (to(self.sky_feather, 7.0), "7"),
        }
        for sp in list(self.trim_a) + list(self.trim_b):
            r[sp] = (to(sp, 0.0), "0")
        self._resets = {w: fn for w, (fn, _) in r.items()}
        for lab in self.findChildren(QLabel):
            field = lab.buddy()
            if field in r:
                lab.installEventFilter(self)
                lab.setToolTip(f"Double-click to reset to {r[field][1]}")

    def _reset_lat(self, spin, attr):
        kind = self.proj_kind.currentData()
        if kind != "equirectangular":          # the full-sphere layout has no latitude limits to reset
            spin.setValue(getattr(Projection.default(kind), attr))

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.MouseButtonDblClick and isinstance(obj, QLabel):
            fn = self._resets.get(obj.buddy())
            if fn is not None:
                fn()
                return True
        return super().eventFilter(obj, ev)

    def _trim_group(self, outer, title, b=False):
        g = QWidget()
        f = QFormLayout(g)
        def trim_slider(tip):
            w = SliderSpin(-180, 180, 0.05, 2, 0.0)       # slider and number box, like the Output view
            w.spin.setToolTip(tip)
            return w

        yaw = trim_slider("Fine yaw trim, degrees")
        pitch = trim_slider("Fine pitch trim, degrees")
        roll = trim_slider("Roll about the lens axis. Use 180 if the camera was flipped upside down." if b
                           else "Roll about the lens axis, degrees")
        f.addRow("Yaw", yaw)
        f.addRow("Pitch", pitch)
        f.addRow("Roll", roll)
        outer.addWidget(self._section(title, g, False, "controls/trim_b" if b else "controls/trim_a"))
        return yaw, pitch, roll

    def _model_changed(self, name):
        for w in self.poly_widgets:
            w.setEnabled(name == "polynomial")
        self._emit()

    def _emit(self, *_):
        self.changed.emit()

    def _pitch_to_tilt(self, v):
        """The Planet / Tunnel slider is the Output view pitch: keep it matching without echoing."""
        self.planet_tilt.spin.blockSignals(True)
        self.planet_tilt.spin.setValue(v)
        self.planet_tilt.spin.blockSignals(False)
        self.planet_tilt.slider.blockSignals(True)
        self.planet_tilt.slider.setValue(int(round(v * 10)))
        self.planet_tilt.slider.blockSignals(False)

    def _tilt_to_pitch(self, v):
        v_now = self.view()
        self.set_view(v_now.yaw, v, v_now.roll)

    # ---- read ----
    def lens_settings(self) -> dict:
        return dict(fov_deg=self.fov.value(), model=self.model.currentText(),
                    poly=(self.k1.value(), self.k2.value(), self.k3.value()),
                    trim_a=tuple(s.value() for s in self.trim_a),
                    trim_b=tuple(s.value() for s in self.trim_b))

    def _sync_lat_enabled(self):
        """The two boxes follow the layout: latitude limits for the flat layouts, one 'reach' box for the planet."""
        kind = self.proj_kind.currentData()
        planet = kind == "planet"
        lo, hi = PLANET_REACH if planet else (5.0, MAX_LAT)
        self.lat_top.blockSignals(True)
        self.lat_top.setRange(lo, hi)
        self.lat_top.blockSignals(False)
        self.lat_top.setEnabled(kind != "equirectangular")
        self.lat_bot.setEnabled(kind not in ("equirectangular", "planet"))
        top_label = self._layout_form.labelForField(self.lat_top)
        bot_label = self._layout_form.labelForField(self.lat_bot)
        top_label.setText("Planet reach" if planet else "Top latitude")
        self.lat_top.setToolTip("How far from the middle of the picture, in degrees, the edge reaches. Bigger "
                                "shows more sky and makes the ground planet smaller." if planet else
                                "Highest latitude shown (degrees above the horizon)")
        for w in (bot_label, self.lat_bot):
            w.setVisible(not planet)
        for w in (self._layout_form.labelForField(self.planet_tilt), self.planet_tilt):
            w.setVisible(planet)

    def _kind_changed(self, _=None):
        """Switching layout loads that layout's usual latitude range. The tiny planet starts looking straight
        down (ground in the middle); leaving it hands the previous view back."""
        kind = self.proj_kind.currentData()
        self._sync_lat_enabled()
        if kind != "equirectangular":
            p = Projection.default(kind)
            for sp, v in ((self.lat_top, p.lat_top_deg), (self.lat_bot, p.lat_bot_deg)):
                sp.blockSignals(True)
                sp.setValue(v)
                sp.blockSignals(False)
        if kind == "planet" and self._view_before_planet is None:
            v = self.view()
            self._view_before_planet = (v.yaw, v.pitch, v.roll)
            self.set_view(v.yaw, -90.0, v.roll)             # also announces the change
            return
        if kind != "planet" and self._view_before_planet is not None:
            yaw, pitch, roll = self._view_before_planet
            self._view_before_planet = None
            if abs(self.view().pitch + 90.0) < 1e-6:        # only if the user left it where we put it
                self.set_view(yaw, pitch, roll)
                return
        self._emit()

    def projection(self) -> Projection:
        kind = self.proj_kind.currentData()
        if kind == "equirectangular":
            return EQUIRECT
        if kind == "planet":
            return Projection(kind, self.lat_top.value(), 0.0)
        return Projection(kind, self.lat_top.value(), self.lat_bot.value())

    def view(self) -> ViewParams:
        return ViewParams(self.v_yaw.value(), self.v_pitch.value(), self.v_roll.value())

    def blend(self) -> BlendParams:
        return BlendParams(
            seam_deg=self.seam.value(), feather_deg=self.feather.value(), sky_feather_deg=self.sky_feather.value(),
            hard_cut=self.hard.isChecked(),
            )   # brightness and white balance are not exposed: both images stay at neutral 1.00

    # ---- save / load ----
    def state_dict(self) -> dict:
        ls, bp, v = self.lens_settings(), self.blend(), self.view()
        return {
            "lens": {"fov_deg": ls["fov_deg"], "model": ls["model"], "poly": list(ls["poly"])},
            "trim_a": list(ls["trim_a"]), "trim_b": list(ls["trim_b"]),
            "view": {"yaw": v.yaw, "pitch": v.pitch, "roll": v.roll},
            "projection": {"kind": self.projection().kind, "lat_top_deg": self.lat_top.value(),
                           "lat_bot_deg": self.lat_bot.value()},
            "blend": {"seam_deg": bp.seam_deg, "feather_deg": bp.feather_deg,
                      "sky_feather_deg": bp.sky_feather_deg, "hard_cut": bp.hard_cut},
        }

    def load_state(self, d: dict):
        """Set every control from a saved dict, then announce one change."""
        widgets = [self.fov, self.k1, self.k2, self.k3, self.seam, self.feather, self.sky_feather, self.model, self.hard,
                   self.v_yaw.spin, self.v_pitch.spin, self.v_roll.spin, self.proj_kind, self.lat_top,
                   self.lat_bot] + list(self.trim_a) + list(self.trim_b)
        for w in widgets:
            w.blockSignals(True)
        try:
            pj = d.get("projection", {})
            kind = pj.get("kind", "equirectangular")
            self.proj_kind.setCurrentIndex(KINDS.index(kind) if kind in KINDS else 0)
            default = Projection.default(self.proj_kind.currentData())
            self._sync_lat_enabled()                          # sets the box range for this layout first
            self.lat_top.setValue(float(pj.get("lat_top_deg", default.lat_top_deg)))
            self.lat_bot.setValue(float(pj.get("lat_bot_deg", default.lat_bot_deg)))
            lens = d.get("lens", {})
            if "fov_deg" in lens:
                self.fov.setValue(float(lens["fov_deg"]))
            if lens.get("model") in MODELS:
                self.model.setCurrentText(lens["model"])
            for sp, v in zip((self.k1, self.k2, self.k3), lens.get("poly", (1.0, 0.0, 0.0))):
                sp.setValue(float(v))
            for sps, key in ((self.trim_a, "trim_a"), (self.trim_b, "trim_b")):
                for sp, v in zip(sps, d.get(key, (0.0, 0.0, 0.0))):
                    sp.setValue(float(v))
            view = d.get("view", {})
            for ss, key in ((self.v_yaw, "yaw"), (self.v_pitch, "pitch"), (self.v_roll, "roll")):
                v = float(view.get(key, 0.0))
                ss.spin.setValue(v)
                ss.slider.blockSignals(True)
                ss.slider.setValue(int(round(v * 10)))
                ss.slider.blockSignals(False)
            b = d.get("blend", {})
            self.seam.setValue(float(b.get("seam_deg", 90.0)))
            self.feather.setValue(float(b.get("feather_deg", 20.0)))
            self.sky_feather.setValue(float(b.get("sky_feather_deg", 7.0)))
            self.hard.setChecked(bool(b.get("hard_cut", False)))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._sync_lat_enabled()
        self._model_changed(self.model.currentText())  # also refreshes the polynomial fields and emits

    # ---- write ----
    def apply_lens_preset(self, fov_deg: float, model: str, trim_a: tuple, trim_b: tuple):
        """Set the lens settings for a known camera (field of view, model and both lens trims); announce once."""
        widgets = [self.fov, self.model] + list(self.trim_a) + list(self.trim_b)
        for w in widgets:
            w.blockSignals(True)
        try:
            self.fov.setValue(float(fov_deg))
            if model in MODELS:
                self.model.setCurrentIndex(MODELS.index(model))
            for spins, vals in ((self.trim_a, trim_a), (self.trim_b, trim_b)):
                for sp, v in zip(spins, vals):
                    sp.setValue(float(v))
        finally:
            for w in widgets:
                w.blockSignals(False)
        self._model_changed(self.model.currentText())      # also announces the change

    def set_trims(self, trim_a: tuple, trim_b: tuple, emit: bool = True):
        """Set both lenses' yaw, pitch, roll trims."""
        spins = list(self.trim_a) + list(self.trim_b)
        for sp in spins:
            sp.blockSignals(True)
        try:
            for sp, v in zip(spins, tuple(trim_a) + tuple(trim_b)):
                sp.setValue(float(v))
        finally:
            for sp in spins:
                sp.blockSignals(False)
        if emit:
            self._emit()

    def set_view(self, yaw: float, pitch: float, roll: float | None = None):
        for ss, v in ((self.v_yaw, yaw), (self.v_pitch, pitch)) + (((self.v_roll, roll),) if roll is not None else ()):
            ss.spin.blockSignals(True)
            ss.spin.setValue(v)
            ss.spin.blockSignals(False)
            ss.slider.blockSignals(True)
            ss.slider.setValue(int(round(v * 10)))
            ss.slider.blockSignals(False)
        self._pitch_to_tilt(pitch)
        self.changed.emit()

    def reset_view(self):
        self.set_view(0.0, 0.0, 0.0)
