"""Export dialog: size, format, layers, metadata, destination."""
from __future__ import annotations

import os

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from .collapsible import CollapsibleSection
from .export import EXTENSIONS, ExportSettings, estimate_memory_mb, exiftool_path
from .reproject import EQUIRECT, KIND_LABELS, Projection, output_size

FORMAT_LABELS = (("tiff", "TIFF (lossless)"), ("png", "PNG (lossless)"), ("jpg", "JPEG"))


class ExportDialog(QDialog):
    def __init__(self, parent, settings: ExportSettings, out_path: str, recommended: int, source_is_16bit: bool,
                 has_exiftool: bool | None = None, proj: Projection = EQUIRECT, has_mask: bool = False):
        super().__init__(parent)
        self.setWindowTitle(f"Export - {KIND_LABELS[proj.kind]}")
        self.setMinimumWidth(560)
        self._recommended = recommended
        self._source_16 = source_is_16bit
        self._proj = proj
        self.has_exiftool = (exiftool_path() is not None) if has_exiftool is None else has_exiftool

        self.width_spin = QSpinBox()
        self.width_spin.setRange(1024, 40000)
        self.width_spin.setSingleStep(2)
        self.width_spin.setValue(settings.width if settings.width else recommended)
        self.rec_btn = QPushButton(f"Recommended ({recommended})")
        self.rec_btn.setToolTip("Matches the detail at the center of your fisheye circles, so nothing is wasted or lost.")
        self.size_label = QLabel()
        self.fmt = QComboBox()
        for key, label in FORMAT_LABELS:
            self.fmt.addItem(label, key)
        self.fmt.setCurrentIndex([k for k, _ in FORMAT_LABELS].index(settings.fmt))
        self.depth16 = QCheckBox("Save as 16 bit (more editing headroom)")
        self.depth16.setChecked(settings.sixteen_bit)
        self.quality = QSpinBox()
        self.quality.setRange(60, 100)
        self.quality.setValue(settings.jpg_quality)
        self.layers = QCheckBox("Also save the two warped layers separately (with transparency)")
        self.layers.setChecked(settings.save_layers)
        self.meta = QCheckBox("Add spherical photo metadata (needs exiftool)")
        meta_ok = self.has_exiftool and proj.is_equirect
        self.meta.setChecked(settings.spherical_metadata and meta_ok)
        self.meta.setEnabled(meta_ok)
        if not self.has_exiftool:
            self.meta.setToolTip("exiftool was not found. Install it (for example: brew install exiftool) to enable this.")
        elif not proj.is_equirect:
            self.meta.setToolTip("Spherical photo metadata only applies to the equirectangular layout.")
        self.use_mask = QCheckBox("Apply the painted seam mask")
        self.use_mask.setChecked(settings.use_mask and has_mask)
        self.use_mask.setEnabled(has_mask)
        if not has_mask:
            self.use_mask.setToolTip("Nothing is painted yet. Use Paint seam on the preview.")
        self.path_edit = QLineEdit(out_path)
        self.browse = QPushButton("Choose...")

        row = QHBoxLayout()
        row.addWidget(self.width_spin)
        row.addWidget(self.rec_btn)
        row.addWidget(self.size_label, 1)
        path_row = QHBoxLayout()
        path_row.addWidget(self.path_edit, 1)
        path_row.addWidget(self.browse)
        form = QFormLayout()
        form.addRow("Width (pixels)", row)
        form.addRow("Format", self.fmt)
        more = QWidget()
        more_form = QFormLayout(more)
        more_form.addRow(self.depth16)
        more_form.addRow("JPEG quality", self.quality)
        more_form.addRow(self.layers)
        more_form.addRow(self.use_mask)
        more_form.addRow(self.meta)
        # Open it straight away when something in it is switched on, so nothing is hidden by surprise.
        unusual = settings.sixteen_bit or settings.save_layers or self.layers.isChecked() or self.depth16.isChecked() \
            or (has_mask and not settings.use_mask) or self.meta.isChecked()
        self.more_section = CollapsibleSection("More options", more, bool(unusual), None)
        form.addRow(self.more_section)
        form.addRow("Save to", path_row)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Export")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

        self.rec_btn.clicked.connect(lambda: self.width_spin.setValue(self._recommended))
        self.width_spin.valueChanged.connect(self._refresh)
        self.fmt.currentIndexChanged.connect(self._fmt_changed)
        self.depth16.toggled.connect(self._refresh)
        self.layers.toggled.connect(self._refresh)
        self.browse.clicked.connect(self._browse)
        self._fmt_changed()

    def _fmt_changed(self, *_):
        key = self.fmt.currentData()
        self.depth16.setEnabled(key != "jpg")
        self.quality.setEnabled(key == "jpg")
        # Keep the extension in step with the format.
        base, _ext = os.path.splitext(self.path_edit.text())
        self.path_edit.setText(base + EXTENSIONS[key])
        self._refresh()

    def _refresh(self, *_):
        w = self.width_spin.value()
        sixteen = (self._source_16 or self.depth16.isChecked()) and self.fmt.currentData() != "jpg"
        _, h = output_size(self._proj, w)
        mb = estimate_memory_mb(w, 2 if sixteen else 1, self.layers.isChecked(), h / w)
        self.size_label.setText(f"{w} x {h}, about {mb / 1000:.1f} GB of memory" if mb > 1000
                                else f"{w} x {h}, about {mb:.0f} MB of memory")

    def _browse(self):
        key = self.fmt.currentData()
        path, _ = QFileDialog.getSaveFileName(self, "Export to", self.path_edit.text(), f"*{EXTENSIONS[key]}")
        if path:
            if not path.lower().endswith(EXTENSIONS[key]):
                path += EXTENSIONS[key]
            self.path_edit.setText(path)

    def settings(self) -> ExportSettings:
        w = self.width_spin.value()
        return ExportSettings(width=w - (w % 2), fmt=self.fmt.currentData(), sixteen_bit=self.depth16.isChecked(),
                              interp="lanczos", jpg_quality=self.quality.value(),
                              save_layers=self.layers.isChecked(),
                              spherical_metadata=self.meta.isChecked() and self.meta.isEnabled(),
                              use_mask=self.use_mask.isChecked() or not self.use_mask.isEnabled())

    def path(self) -> str:
        return self.path_edit.text().strip()
