"""FFT analysis tab — select a namespace array, compute spectrum, apply filters."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from pyside_app.plot_view import PlotView

_LABEL_SS = "color:#001f41; font-size:12px; font-weight:700;"
_TEXT_SS = "color:#355070; font-size:12px;"
_BTN_PRIMARY = (
    "QPushButton { background:#001f41; color:white; border-radius:6px; padding:4px 14px;"
    " font-weight:600; min-height:28px; } QPushButton:hover { background:#0d3567; }"
)
_BTN_SECONDARY = (
    "QPushButton { background:#e2e8f0; color:#0f1b2b; border-radius:6px; padding:4px 14px;"
    " min-height:28px; } QPushButton:hover { background:#cbd5e1; }"
)
_COMBO_SS = (
    "QComboBox { background:#fff; border:1px solid #d1dce8; border-radius:6px;"
    " padding:4px 8px; font-size:12px; color:#0f1b2b; min-height:28px; }"
    "QComboBox QAbstractItemView { selection-background-color:#c7def5; color:#0f1b2b; }"
)
_SLIDER_SS = (
    "QSlider::groove:horizontal { height:6px; border-radius:3px; background:#cbd5e1; }"
    "QSlider::sub-page:horizontal { border-radius:3px; background:#001f41; }"
    "QSlider::handle:horizontal { width:16px; margin:-5px 0; border-radius:8px; background:#b60021; }"
)


class FFTTab(QWidget):
    """Interactive FFT analysis panel wired to the notebook namespace."""

    def __init__(self, graph_state: Any = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        print("[debug][fft-tab] init:start", flush=True)
        self._namespace: dict[str, Any] = {}
        self._graph_state = graph_state

        if graph_state is not None:
            graph_state.namespace_changed.connect(self._on_namespace_changed)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("FFT / Spectral Analysis", self)
        title.setStyleSheet("color:#001f41; font-size:15px; font-weight:700;")
        root.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setStyleSheet("QScrollArea { border:none; background:#f0f4f8; }")
        controls_widget = QWidget()
        controls_widget.setStyleSheet("background:#f0f4f8;")
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(12)
        controls_scroll.setWidget(controls_widget)

        # ── Signal selection ──────────────────────────────────────────────────
        sig_frame = self._section_frame("Signal")
        sig_form = QFormLayout()
        sig_form.setSpacing(6)

        self.array_combo = QComboBox(sig_frame)
        self.array_combo.setStyleSheet(_COMBO_SS)
        self.array_combo.setPlaceholderText("Select array from namespace…")
        sig_form.addRow(QLabel("Array (y):", sig_frame), self.array_combo)

        self.x_combo = QComboBox(sig_frame)
        self.x_combo.setStyleSheet(_COMBO_SS)
        self.x_combo.addItem("Auto (index)", "__auto__")
        sig_form.addRow(QLabel("x / time:", sig_frame), self.x_combo)

        self.fs_spin = QDoubleSpinBox(sig_frame)
        self.fs_spin.setRange(1e-9, 1e12)
        self.fs_spin.setValue(1.0)
        self.fs_spin.setDecimals(4)
        self.fs_spin.setSuffix("  Hz")
        self.fs_spin.setStyleSheet(
            "QDoubleSpinBox { border:1px solid #d1dce8; border-radius:6px; padding:4px 6px;"
            " font-size:12px; color:#0f1b2b; background:#fff; min-height:28px; }"
        )
        sig_form.addRow(QLabel("Sample rate:", sig_frame), self.fs_spin)
        sig_frame.layout().addLayout(sig_form)
        controls_layout.addWidget(sig_frame)

        # ── Display options ───────────────────────────────────────────────────
        disp_frame = self._section_frame("Display")
        disp_form = QFormLayout()
        disp_form.setSpacing(6)

        self.scale_combo = QComboBox(disp_frame)
        self.scale_combo.setStyleSheet(_COMBO_SS)
        for label, val in [("Magnitude (linear)", "magnitude"), ("Power (linear)", "power"),
                            ("Power dB", "db"), ("Phase (rad)", "phase")]:
            self.scale_combo.addItem(label, val)
        disp_form.addRow(QLabel("Scale:", disp_frame), self.scale_combo)

        self.onesided_combo = QComboBox(disp_frame)
        self.onesided_combo.setStyleSheet(_COMBO_SS)
        self.onesided_combo.addItem("One-sided (0…Nyquist)", True)
        self.onesided_combo.addItem("Two-sided", False)
        disp_form.addRow(QLabel("Sides:", disp_frame), self.onesided_combo)

        self.window_combo = QComboBox(disp_frame)
        self.window_combo.setStyleSheet(_COMBO_SS)
        for w in ["none", "hann", "hamming", "blackman", "flattop"]:
            self.window_combo.addItem(w.title(), w)
        disp_form.addRow(QLabel("Window:", disp_frame), self.window_combo)
        disp_frame.layout().addLayout(disp_form)
        controls_layout.addWidget(disp_frame)

        # ── Filter ────────────────────────────────────────────────────────────
        filt_frame = self._section_frame("Filter (Butterworth)")
        filt_form = QFormLayout()
        filt_form.setSpacing(6)

        self.filter_combo = QComboBox(filt_frame)
        self.filter_combo.setStyleSheet(_COMBO_SS)
        for label, val in [("None", "none"), ("Low-pass", "low"), ("High-pass", "high"), ("Band-pass", "band")]:
            self.filter_combo.addItem(label, val)
        self.filter_combo.currentIndexChanged.connect(self._sync_filter_ui)
        filt_form.addRow(QLabel("Type:", filt_frame), self.filter_combo)

        self.cutoff_low_spin = QDoubleSpinBox(filt_frame)
        self.cutoff_low_spin.setRange(0, 1e9)
        self.cutoff_low_spin.setValue(100.0)
        self.cutoff_low_spin.setSuffix("  Hz")
        self.cutoff_low_spin.setStyleSheet(
            "QDoubleSpinBox { border:1px solid #d1dce8; border-radius:6px; padding:4px 6px;"
            " font-size:12px; color:#0f1b2b; background:#fff; min-height:28px; }"
        )
        self._cutoff_low_label = QLabel("Cutoff:", filt_frame)
        filt_form.addRow(self._cutoff_low_label, self.cutoff_low_spin)

        self.cutoff_high_spin = QDoubleSpinBox(filt_frame)
        self.cutoff_high_spin.setRange(0, 1e9)
        self.cutoff_high_spin.setValue(500.0)
        self.cutoff_high_spin.setSuffix("  Hz")
        self.cutoff_high_spin.setStyleSheet(
            "QDoubleSpinBox { border:1px solid #d1dce8; border-radius:6px; padding:4px 6px;"
            " font-size:12px; color:#0f1b2b; background:#fff; min-height:28px; }"
        )
        self._cutoff_high_row_label = QLabel("High cutoff:", filt_frame)
        filt_form.addRow(self._cutoff_high_row_label, self.cutoff_high_spin)

        self.order_spin = QDoubleSpinBox(filt_frame)
        self.order_spin.setRange(1, 10)
        self.order_spin.setValue(4)
        self.order_spin.setDecimals(0)
        self.order_spin.setStyleSheet(
            "QDoubleSpinBox { border:1px solid #d1dce8; border-radius:6px; padding:4px 6px;"
            " font-size:12px; color:#0f1b2b; background:#fff; min-height:28px; }"
        )
        filt_form.addRow(QLabel("Order:", filt_frame), self.order_spin)
        filt_frame.layout().addLayout(filt_form)
        controls_layout.addWidget(filt_frame)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self.compute_btn = QPushButton("Compute FFT", self)
        self.compute_btn.setStyleSheet(_BTN_PRIMARY)
        self.compute_btn.clicked.connect(self._compute)
        btn_row.addWidget(self.compute_btn)
        self.refresh_btn = QPushButton("Refresh Arrays", self)
        self.refresh_btn.setStyleSheet(_BTN_SECONDARY)
        self.refresh_btn.clicked.connect(self._reload_array_combos)
        btn_row.addWidget(self.refresh_btn)
        controls_layout.addLayout(btn_row)

        self.status_label = QLabel("Select an array and press Compute FFT.", self)
        self.status_label.setStyleSheet(_TEXT_SS)
        self.status_label.setWordWrap(True)
        controls_layout.addWidget(self.status_label)

        self.dominant_label = QLabel("", self)
        self.dominant_label.setStyleSheet("color:#001f41; font-size:11px; font-weight:600;")
        self.dominant_label.setWordWrap(True)
        controls_layout.addWidget(self.dominant_label)

        controls_layout.addStretch(1)

        splitter.addWidget(controls_scroll)

        # ── Plot area ─────────────────────────────────────────────────────────
        plot_area = QWidget()
        plot_area.setStyleSheet("background:#ffffff;")
        plot_layout = QVBoxLayout(plot_area)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setSpacing(4)

        self.spectrum_plot = PlotView(plot_area)
        plot_layout.addWidget(self.spectrum_plot, 3)

        self.signal_plot = PlotView(plot_area)
        plot_layout.addWidget(self.signal_plot, 2)

        splitter.addWidget(plot_area)
        splitter.setSizes([320, 900])
        root.addWidget(splitter, 1)

        self._sync_filter_ui()
        self._show_welcome()
        print("[debug][fft-tab] init:done", flush=True)

    # ── UI helpers ────────────────────────────────────────────────────────────

    def _section_frame(self, title: str) -> QFrame:
        frame = QFrame(self)
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet(
            "QFrame { background:#ffffff; border:1px solid #d1dce8; border-radius:6px; }"
            "QLabel { background:transparent; border:none; color:#334155; font-size:12px; }"
        )
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        lbl = QLabel(title, frame)
        lbl.setStyleSheet("font-weight:700; color:#001f41; font-size:12px; background:transparent; border:none;")
        layout.addWidget(lbl)
        return frame

    def _sync_filter_ui(self) -> None:
        ftype = self.filter_combo.currentData()
        band = ftype == "band"
        self._cutoff_low_label.setText("Low cutoff:" if band else "Cutoff:")
        self.cutoff_high_spin.setVisible(band)
        self._cutoff_high_row_label.setVisible(band)
        self.cutoff_low_spin.setEnabled(ftype != "none")
        self.order_spin.setEnabled(ftype != "none")

    def _show_welcome(self) -> None:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            title="FFT Spectrum (no data)",
            annotations=[{"text": "Select an array and press Compute FFT", "showarrow": False,
                           "font": {"size": 14, "color": "#64748b"}, "xref": "paper", "yref": "paper",
                           "x": 0.5, "y": 0.5}],
        )
        self.spectrum_plot.set_figure(fig)
        self.signal_plot.set_figure(go.Figure(layout=go.Layout(template="plotly_white", title="Signal (no data)")))

    # ── Namespace sync ────────────────────────────────────────────────────────

    def _on_namespace_changed(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace
        self._reload_array_combos()

    def _reload_array_combos(self) -> None:
        print("[debug][fft-tab] reload_array_combos", flush=True)
        arrays = [
            (name, val) for name, val in self._namespace.items()
            if isinstance(val, np.ndarray) and val.ndim == 1 and val.size >= 4
            and not name.startswith("_")
        ]
        arrays.sort(key=lambda t: t[0])

        current_y = self.array_combo.currentData()
        current_x = self.x_combo.currentData()

        self.array_combo.blockSignals(True)
        self.x_combo.blockSignals(True)
        self.array_combo.clear()
        self.x_combo.clear()
        self.x_combo.addItem("Auto (index)", "__auto__")

        for name, _val in arrays:
            self.array_combo.addItem(name, name)
            self.x_combo.addItem(name, name)

        idx_y = self.array_combo.findData(current_y)
        if idx_y >= 0:
            self.array_combo.setCurrentIndex(idx_y)
        idx_x = self.x_combo.findData(current_x)
        if idx_x >= 0:
            self.x_combo.setCurrentIndex(idx_x)

        self.array_combo.blockSignals(False)
        self.x_combo.blockSignals(False)

    # ── Computation ───────────────────────────────────────────────────────────

    def _compute(self) -> None:
        y_name = self.array_combo.currentData()
        if y_name is None:
            self.status_label.setText("No array selected.")
            return

        y_raw = self._namespace.get(y_name)
        if not isinstance(y_raw, np.ndarray):
            self.status_label.setText(f"'{y_name}' is not a numpy array.")
            return

        y = y_raw.ravel().astype(float)
        n = len(y)

        x_data_key = self.x_combo.currentData()
        if x_data_key == "__auto__":
            fs = float(self.fs_spin.value())
            t = np.arange(n) / fs
        else:
            t_raw = self._namespace.get(x_data_key)
            if isinstance(t_raw, np.ndarray) and t_raw.size == n:
                t = t_raw.ravel().astype(float)
                dt = float(np.mean(np.diff(t))) if n > 1 else 1.0
                fs = 1.0 / dt if dt > 0 else 1.0
                self.fs_spin.setValue(fs)
            else:
                t = np.arange(n)
                fs = float(self.fs_spin.value())

        # Apply window
        window_name = self.window_combo.currentData() or "none"
        if window_name == "hann":
            window = np.hanning(n)
        elif window_name == "hamming":
            window = np.hamming(n)
        elif window_name == "blackman":
            window = np.blackman(n)
        elif window_name == "flattop":
            from scipy.signal.windows import flattop
            window = flattop(n)
        else:
            window = np.ones(n)

        y_windowed = y * window

        # Apply filter
        y_filtered = self._apply_filter(y, fs)

        # Compute FFT on filtered signal
        y_for_fft = self._apply_filter(y_windowed, fs)
        fft_vals = np.fft.rfft(y_for_fft)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        scale = self.scale_combo.currentData() or "magnitude"
        onesided = self.onesided_combo.currentData()

        if not onesided:
            fft_full = np.fft.fft(y_for_fft)
            freqs_full = np.fft.fftfreq(n, d=1.0 / fs)
            sort_idx = np.argsort(freqs_full)
            freqs_plot = freqs_full[sort_idx]
            spectrum = np.abs(fft_full[sort_idx]) / n
        else:
            freqs_plot = freqs
            spectrum = np.abs(fft_vals) / n

        if scale == "power":
            y_spec = spectrum ** 2
            y_label = "Power"
        elif scale == "db":
            y_spec = 20 * np.log10(np.maximum(spectrum, 1e-300))
            y_label = "Power (dB)"
        elif scale == "phase":
            y_spec = np.angle(fft_vals if onesided else np.fft.fft(y_for_fft))
            if not onesided:
                y_spec = y_spec[sort_idx]
            y_label = "Phase (rad)"
        else:
            y_spec = spectrum
            y_label = "Magnitude"

        # Dominant frequencies (top 5)
        mag = np.abs(fft_vals) / n
        top_idx = np.argsort(mag)[::-1][:5]
        dom_parts = [f"{freqs[i]:.4g} Hz (|A|={mag[i]:.3g})" for i in top_idx if freqs[i] > 0]
        self.dominant_label.setText("Dominant: " + ",  ".join(dom_parts[:3]))

        # Spectrum figure
        spec_fig = go.Figure()
        spec_fig.add_trace(go.Scatter(
            x=freqs_plot, y=y_spec,
            mode="lines",
            name=f"FFT[{y_name}]",
            line={"color": "#001f41", "width": 1.5},
        ))
        spec_fig.update_layout(
            template="plotly_white",
            title=f"Spectrum — {y_name} ({window_name} window)",
            xaxis_title="Frequency (Hz)",
            yaxis_title=y_label,
            margin={"l": 50, "r": 20, "t": 40, "b": 40},
        )
        self.spectrum_plot.set_figure(spec_fig)

        # Signal figure (original + filtered)
        sig_fig = go.Figure()
        sig_fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=y_name,
                                     line={"color": "#94a3b8", "width": 1}))
        if y_filtered is not None:
            sig_fig.add_trace(go.Scatter(x=t, y=y_filtered, mode="lines", name="Filtered",
                                          line={"color": "#b60021", "width": 2}))
        sig_fig.update_layout(
            template="plotly_white",
            title="Signal",
            xaxis_title="Time" if x_data_key == "__auto__" else x_data_key,
            yaxis_title=y_name,
            margin={"l": 50, "r": 20, "t": 40, "b": 40},
        )
        self.signal_plot.set_figure(sig_fig)
        self.status_label.setText(
            f"Computed FFT of '{y_name}'  ·  N={n}  ·  fs={fs:.4g} Hz  ·  Δf={fs/n:.4g} Hz"
        )
        print(f"[debug][fft-tab] computed y={y_name!r} n={n} fs={fs}", flush=True)

    def _apply_filter(self, y: np.ndarray, fs: float) -> np.ndarray | None:
        ftype = self.filter_combo.currentData()
        if ftype == "none" or ftype is None:
            return None
        try:
            from scipy.signal import butter, filtfilt
            order = int(self.order_spin.value())
            nyq = fs / 2.0
            if ftype == "band":
                lo = float(self.cutoff_low_spin.value()) / nyq
                hi = float(self.cutoff_high_spin.value()) / nyq
                lo = max(1e-6, min(lo, 0.9999))
                hi = max(lo + 1e-6, min(hi, 0.9999))
                b, a = butter(order, [lo, hi], btype="bandpass")
            elif ftype == "high":
                wn = max(1e-6, min(float(self.cutoff_low_spin.value()) / nyq, 0.9999))
                b, a = butter(order, wn, btype="high")
            else:
                wn = max(1e-6, min(float(self.cutoff_low_spin.value()) / nyq, 0.9999))
                b, a = butter(order, wn, btype="low")
            return filtfilt(b, a, y)
        except Exception as exc:
            print(f"[debug][fft-tab] filter error={exc!r}", flush=True)
            self.status_label.setText(f"Filter error: {exc}")
            return None

    def set_namespace(self, namespace: dict[str, Any]) -> None:
        self._namespace = namespace
        self._reload_array_combos()
