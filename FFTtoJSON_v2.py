"""
FFT Seismic Analyzer  —  Patrick's Earthquake Engineering Lab
=============================================================
Features:
  1. Konno-Ohmachi Smoothing
  2. HVSR (Horizontal-to-Vertical Spectral Ratio)
  3. Bandpass Filter
  5. Arias Intensity
  6. Spectrogram
  8. Multi-file Overlay
  9. Normalized Acceleration Plot (÷PGA) ±1
"""

import sys, os
import numpy as np
from scipy import signal as scipy_signal
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks as _fp
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.ticker as ticker
import matplotlib.cm as cm

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QGroupBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QTextEdit, QStatusBar,
    QFrame, QCheckBox, QMessageBox, QListWidget, QListWidgetItem,
    QSplitter, QScrollArea, QGridLayout, QAbstractItemView
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont

# ── Palette ─────────────────────────────────────────────────────────────────
DARK_BG  = "#0d1117"
PANEL_BG = "#161b22"
CARD_BG  = "#21262d"
ACCENT   = "#58a6ff"
ACCENT2  = "#3fb950"
ACCENT3  = "#ff7b72"
ACCENT4  = "#d2a8ff"
ACCENT5  = "#ffa657"
TEXT_PRI = "#e6edf3"
TEXT_SEC = "#8b949e"
BORDER   = "#30363d"

OVERLAY_COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4, ACCENT5,
                  "#79c0ff", "#56d364", "#ffa198", "#bc8cff", "#ffb77c"]

MPLSTYLE = {
    "axes.facecolor":   DARK_BG,
    "figure.facecolor": PANEL_BG,
    "axes.edgecolor":   BORDER,
    "axes.labelcolor":  TEXT_PRI,
    "xtick.color":      TEXT_SEC,
    "ytick.color":      TEXT_SEC,
    "text.color":       TEXT_PRI,
    "grid.color":       CARD_BG,
    "grid.alpha":       0.9,
    "axes.grid":        True,
    "font.family":      "monospace",
    "axes.titlesize":   10,
    "axes.labelsize":   9,
}

STYLESHEET = f"""
QMainWindow, QWidget {{ background:{DARK_BG}; color:{TEXT_PRI};
    font-family:'Consolas',monospace; font-size:12px; }}
QGroupBox {{ border:1px solid {BORDER}; border-radius:6px; margin-top:10px;
    padding-top:8px; color:{TEXT_SEC}; font-size:10px; }}
QGroupBox::title {{ subcontrol-origin:margin; left:10px;
    padding:0 4px; color:{ACCENT}; font-size:11px; }}
QPushButton {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:5px;
    padding:5px 12px; color:{TEXT_PRI}; }}
QPushButton:hover {{ background:{ACCENT}; color:#000; border-color:{ACCENT}; }}
QPushButton#danger {{ color:{ACCENT3}; border-color:{ACCENT3}; }}
QPushButton#danger:hover {{ background:{ACCENT3}; color:#000; }}
QPushButton#success {{ color:{ACCENT2}; border-color:{ACCENT2}; }}
QPushButton#success:hover {{ background:{ACCENT2}; color:#000; }}
QPushButton#purple {{ color:{ACCENT4}; border-color:{ACCENT4}; }}
QPushButton#purple:hover {{ background:{ACCENT4}; color:#000; }}
QComboBox {{ background:{CARD_BG}; border:1px solid {BORDER}; border-radius:4px;
    padding:3px 8px; color:{TEXT_PRI}; }}
QComboBox QAbstractItemView {{ background:{CARD_BG}; color:{TEXT_PRI};
    selection-background-color:{ACCENT}; }}
QLabel {{ color:{TEXT_PRI}; }}
QSpinBox, QDoubleSpinBox {{ background:{CARD_BG}; border:1px solid {BORDER};
    border-radius:4px; padding:3px 6px; color:{TEXT_PRI}; }}
QTextEdit {{ background:{CARD_BG}; border:1px solid {BORDER};
    border-radius:4px; color:{TEXT_PRI}; font-family:monospace; font-size:11px; }}
QTabWidget::pane {{ border:1px solid {BORDER}; background:{PANEL_BG}; }}
QTabBar::tab {{ background:{CARD_BG}; color:{TEXT_SEC};
    padding:6px 16px; border-bottom:2px solid transparent; }}
QTabBar::tab:selected {{ color:{ACCENT}; border-bottom:2px solid {ACCENT};
    background:{PANEL_BG}; }}
QListWidget {{ background:{CARD_BG}; border:1px solid {BORDER};
    border-radius:4px; color:{TEXT_PRI}; }}
QListWidget::item:selected {{ background:{ACCENT}; color:#000; }}
QCheckBox {{ color:{TEXT_PRI}; spacing:6px; }}
QStatusBar {{ background:{PANEL_BG}; color:{TEXT_SEC};
    border-top:1px solid {BORDER}; }}
QScrollArea {{ border:none; }}
QSplitter::handle {{ background:{BORDER}; }}
"""


# ════════════════════════════════════════════════════════════════════════════
#  DSP Functions
# ════════════════════════════════════════════════════════════════════════════

def apply_bandpass(data, fs, f_low, f_high, order=4):
    nyq = fs / 2.0
    if f_low <= 0 or f_high >= nyq or f_low >= f_high:
        return data
    b, a = scipy_signal.butter(order, [f_low/nyq, f_high/nyq], btype='band')
    return scipy_signal.filtfilt(b, a, data)


def compute_fft(data, fs, window="hann", ymode="Amplitude (Linear)", one_sided=True):
    N = len(data)
    if N == 0:
        return np.array([]), np.array([])
    wins = {"hann": np.hanning, "hamming": np.hamming,
            "blackman": np.blackman, "rectangular": np.ones}
    win = wins.get(window, np.hanning)(N)
    win_scale = np.sum(win)
    Y = np.fft.fft(data * win)
    freqs = np.fft.fftfreq(N, d=1.0/fs)
    if one_sided:
        h = N // 2
        freqs = freqs[:h]
        amp = (2.0 / win_scale) * np.abs(Y[:h])
    else:
        freqs = np.fft.fftshift(freqs)
        amp = (1.0 / win_scale) * np.abs(np.fft.fftshift(Y))
    if ymode == "Amplitude (dB)":
        amp = 20 * np.log10(np.maximum(amp, 1e-12))
    elif ymode == "Power Spectral Density":
        amp = (amp ** 2) / (fs * win_scale)
        amp = np.maximum(amp, 1e-30)
    return freqs, amp


def konno_ohmachi_smooth(freqs, amp, b=40.0):
    n = len(freqs)
    smoothed = np.zeros(n)
    for i, fc in enumerate(freqs):
        if fc <= 0:
            smoothed[i] = amp[i]
            continue
        ratio = freqs / fc
        ratio = np.where(ratio <= 0, 1e-10, ratio)
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = b * np.log10(ratio)
            w = np.where(np.abs(arg) < 1e-6, 1.0,
                         (np.sin(arg) / arg) ** 4)
        w[freqs <= 0] = 0
        total = w.sum()
        smoothed[i] = (w * amp).sum() / total if total > 0 else amp[i]
    return smoothed


def compute_hvsr(ns, ew, ud, fs, window="hann", smooth_b=40.0, one_sided=True):
    f1, a_ns = compute_fft(ns, fs, window=window, ymode="Amplitude (Linear)", one_sided=one_sided)
    _,  a_ew = compute_fft(ew, fs, window=window, ymode="Amplitude (Linear)", one_sided=one_sided)
    _,  a_ud = compute_fft(ud, fs, window=window, ymode="Amplitude (Linear)", one_sided=one_sided)
    a_ns = konno_ohmachi_smooth(f1, a_ns, b=smooth_b)
    a_ew = konno_ohmachi_smooth(f1, a_ew, b=smooth_b)
    a_ud = konno_ohmachi_smooth(f1, a_ud, b=smooth_b)
    a_ud = np.maximum(a_ud, 1e-12)
    hvsr = np.sqrt(a_ns**2 + a_ew**2) / a_ud
    return f1, hvsr


def compute_arias(data, fs):
    g = 9.81
    dt = 1.0 / fs
    a2 = data ** 2
    ia = cumulative_trapezoid(a2, dx=dt, initial=0) * (np.pi / (2 * g))
    return ia


def find_peaks_fft(freqs, amp, n=5):
    if len(amp) == 0:
        return [], []
    idx, _ = scipy_signal.find_peaks(amp, height=np.percentile(amp, 75), distance=5)
    if len(idx) == 0:
        return [], []
    top = idx[np.argsort(amp[idx])[::-1]][:n]
    return freqs[top], amp[top]


# ════════════════════════════════════════════════════════════════════════════
#  Canvas helper
# ════════════════════════════════════════════════════════════════════════════

class PlotCanvas(QWidget):
    def __init__(self, nrows=1, ncols=1, figsize=(10, 5), parent=None):
        super().__init__(parent)
        plt.rcParams.update(MPLSTYLE)
        self.fig = Figure(figsize=figsize, tight_layout=True)
        self.axes = []
        for i in range(nrows * ncols):
            ax = self.fig.add_subplot(nrows, ncols, i+1)
            self._style_ax(ax)
            self.axes.append(ax)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setStyleSheet(f"background:{PANEL_BG}; color:{TEXT_PRI};")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0,0,0,0)
        lay.addWidget(self.toolbar)
        lay.addWidget(self.canvas)

    def _style_ax(self, ax):
        ax.set_facecolor(DARK_BG)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)

    def clear_all(self):
        for ax in self.axes:
            ax.cla()
            self._style_ax(ax)

    def draw(self):
        self.canvas.draw_idle()


import matplotlib.pyplot as plt


def _smart_genfromtxt(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                delim = ',' if ',' in line else None
                break
        else:
            delim = None
    raw = np.genfromtxt(path, delimiter=delim, comments='#',
                        invalid_raise=False, encoding=None)
    if raw.ndim == 1:
        if delim is None and np.isnan(raw).all():
            raw = np.genfromtxt(path, delimiter=',', comments='#',
                                invalid_raise=False, encoding=None)
        raw = raw[~np.isnan(raw)].reshape(-1, 1) if raw.ndim == 1 else raw
    else:
        raw = raw[~np.isnan(raw).all(axis=1)]
    return raw


# ════════════════════════════════════════════════════════════════════════════
#  File Import Dialog
# ════════════════════════════════════════════════════════════════════════════

class FileImportDialog(QWidget):
    confirmed = pyqtSignal(float, int)

    def __init__(self, filename, ncols, parent=None):
        super().__init__(parent, Qt.Window)
        self.setWindowTitle("Import Signal File")
        self.setFixedSize(500, 420)
        self.setWindowModality(Qt.ApplicationModal)
        self.setStyleSheet(f"""
            QWidget {{ background:{DARK_BG}; color:{TEXT_PRI};
                       font-family:'Consolas',monospace; font-size:13px; }}
            QGroupBox {{ border:1px solid {BORDER}; border-radius:6px;
                margin-top:10px; padding-top:8px; color:{TEXT_SEC}; }}
            QGroupBox::title {{ subcontrol-origin:margin; left:10px;
                padding:0 4px; color:{ACCENT}; font-size:12px; }}
            QDoubleSpinBox, QSpinBox {{
                background:{CARD_BG}; border:2px solid {ACCENT};
                border-radius:5px; padding:8px 10px;
                color:{TEXT_PRI}; font-size:18px;
                min-height:42px; min-width:200px;
            }}
            QDoubleSpinBox::up-button, QSpinBox::up-button {{
                width:28px; border-left:1px solid {BORDER};
            }}
            QDoubleSpinBox::down-button, QSpinBox::down-button {{
                width:28px; border-left:1px solid {BORDER};
            }}
            QPushButton {{ background:{CARD_BG}; border:1px solid {BORDER};
                border-radius:5px; padding:8px 14px; color:{TEXT_PRI};
                font-size:13px; }}
            QPushButton:hover {{ background:{ACCENT}; color:#000; }}
            QLabel {{ color:{TEXT_PRI}; }}
        """)

        lay = QVBoxLayout(self)
        lay.setSpacing(8)
        lay.setContentsMargins(20, 14, 20, 14)

        lbl_fn = QLabel(f"<b>File:</b> {filename}")
        lbl_nc = QLabel(f"<b>Columns detected:</b> {ncols}")
        for lb in (lbl_fn, lbl_nc):
            lb.setStyleSheet(f"color:{TEXT_PRI}; font-size:13px;")
        lay.addWidget(lbl_fn)
        lay.addWidget(lbl_nc)

        if ncols == 1:   hint, hc = "1 col  ->  [ACC]", ACCENT
        elif ncols == 2: hint, hc = "2 cols  ->  [Time | ACC]  (single)", ACCENT2
        elif ncols == 3: hint, hc = "3 cols  ->  [EW | NS | V]  (3 components, no time)", ACCENT4
        else:            hint, hc = f"{ncols} cols  ->  [Time | EW | NS | V ...]", ACCENT5

        lbl_hint = QLabel(hint)
        lbl_hint.setStyleSheet(f"color:{hc}; font-size:13px; font-weight:bold;")
        lay.addWidget(lbl_hint)

        sep = QFrame(); sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background:{BORDER}; max-height:1px")
        lay.addWidget(sep)

        g1 = QGroupBox("  Time Step  (dt)")
        v1 = QVBoxLayout(g1); v1.setSpacing(4); v1.setContentsMargins(10,14,10,10)
        lbl_dt = QLabel("dt (seconds)  e.g.  0.005 = 200 Hz,   0.01 = 100 Hz")
        lbl_dt.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px;")

        row_dt = QHBoxLayout(); row_dt.setSpacing(10)
        self.spin_dt = QDoubleSpinBox()
        self.spin_dt.setRange(0.0001, 100)
        self.spin_dt.setDecimals(4)
        self.spin_dt.setValue(0.005)
        self.spin_dt.setLocale(__import__('PyQt5.QtCore', fromlist=['QLocale']).QLocale(
            __import__('PyQt5.QtCore', fromlist=['QLocale']).QLocale.English))

        self.hz_lbl = QLabel("= 200.00 Hz")
        self.hz_lbl.setStyleSheet(
            f"color:{ACCENT}; font-size:16px; font-weight:bold; padding-left:8px")
        self.hz_lbl.setMinimumWidth(130)
        self.spin_dt.valueChanged.connect(
            lambda v: self.hz_lbl.setText(f"= {1/v:.2f} Hz" if v > 0 else ""))

        row_dt.addWidget(self.spin_dt, stretch=1)
        row_dt.addWidget(self.hz_lbl, stretch=0)
        v1.addWidget(lbl_dt)
        v1.addLayout(row_dt)
        lay.addWidget(g1)

        g2 = QGroupBox("  Acceleration Start Column  (0-based)")
        v2 = QVBoxLayout(g2); v2.setSpacing(4); v2.setContentsMargins(10,14,10,10)
        lbl_col = QLabel("Column index of first acceleration  (0 = leftmost)")
        lbl_col.setStyleSheet(f"color:{TEXT_SEC}; font-size:11px;")

        self.spin_col = QSpinBox()
        self.spin_col.setRange(0, 99)
        self.spin_col.setValue(1 if ncols >= 2 else 0)
        self.spin_col.setLocale(__import__('PyQt5.QtCore', fromlist=['QLocale']).QLocale(
            __import__('PyQt5.QtCore', fromlist=['QLocale']).QLocale.English))

        v2.addWidget(lbl_col)
        v2.addWidget(self.spin_col)
        lay.addWidget(g2)

        hb = QHBoxLayout(); hb.setSpacing(10)
        btn_ok = QPushButton("  Import")
        btn_ok.setMinimumHeight(40)
        btn_ok.setStyleSheet(f"color:{ACCENT2}; border-color:{ACCENT2}; font-size:14px; font-weight:bold;")
        btn_ok.clicked.connect(self._confirm)
        btn_cancel = QPushButton("  Cancel")
        btn_cancel.setMinimumHeight(40)
        btn_cancel.setStyleSheet(f"color:{ACCENT3}; border-color:{ACCENT3}; font-size:14px;")
        btn_cancel.clicked.connect(self.close)
        hb.addWidget(btn_ok); hb.addWidget(btn_cancel)
        lay.addLayout(hb)

    def _confirm(self):
        self.confirmed.emit(self.spin_dt.value(), self.spin_col.value())
        self.close()


# ════════════════════════════════════════════════════════════════════════════
#  Per-component Filter Panel
# ════════════════════════════════════════════════════════════════════════════

class ComponentFilterPanel(QGroupBox):
    def __init__(self, comp_name, color, parent=None):
        super().__init__(f"🔁  {comp_name} — HP/LP Filter + IFFT", parent)
        self.comp_name = comp_name
        self.color     = color
        self._data     = None
        self._fs       = None
        self._t        = None
        self._modified = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setSpacing(4)

        row1 = QHBoxLayout()
        self.chk_hp = QCheckBox("HP")
        self.chk_hp.setChecked(True)
        self.chk_hp.setStyleSheet(f"color:{ACCENT3}; font-weight:bold;")
        self.spin_hp = QDoubleSpinBox()
        self.spin_hp.setRange(0.001, 1000); self.spin_hp.setValue(0.1)
        self.spin_hp.setDecimals(3)
        self.spin_hp.setSuffix(" Hz"); self.spin_hp.setFixedWidth(100)

        self.chk_lp = QCheckBox("LP")
        self.chk_lp.setChecked(True)
        self.chk_lp.setStyleSheet(f"color:{ACCENT4}; font-weight:bold;")
        self.spin_lp = QDoubleSpinBox()
        self.spin_lp.setRange(0.001, 10000); self.spin_lp.setValue(25.0)
        self.spin_lp.setDecimals(3)
        self.spin_lp.setSuffix(" Hz"); self.spin_lp.setFixedWidth(100)

        for w in (self.chk_hp, self.spin_hp,
                  QLabel("  "), self.chk_lp, self.spin_lp):
            row1.addWidget(w)
        row1.addStretch()
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        lbl_t = QLabel("Taper:")
        self.spin_taper = QDoubleSpinBox()
        self.spin_taper.setRange(0, 100); self.spin_taper.setValue(0.5)
        self.spin_taper.setDecimals(3)
        self.spin_taper.setSuffix(" Hz"); self.spin_taper.setFixedWidth(90)

        self.btn_preview = QPushButton("👁 Preview Spectrum")
        self.btn_preview.setObjectName("success")
        self.btn_preview.clicked.connect(self._preview)

        self.btn_ifft = QPushButton("🔁 Convert → Waveform")
        self.btn_ifft.setObjectName("purple")
        self.btn_ifft.setEnabled(False)
        self.btn_ifft.clicked.connect(self._run_ifft)

        for w in (lbl_t, self.spin_taper,
                  self.btn_preview, self.btn_ifft):
            row2.addWidget(w)
        row2.addStretch()
        lay.addLayout(row2)

        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        lay.addWidget(self.lbl_status)

    def set_data(self, data, fs, t):
        self._data = data
        self._fs   = fs
        self._t    = t
        self._modified = None
        self.btn_ifft.setEnabled(False)
        self.lbl_status.setText("")

    def _build_mask(self, freqs):
        mask  = np.ones(len(freqs))
        taper = self.spin_taper.value()
        if self.chk_hp.isChecked():
            f_hp    = self.spin_hp.value()
            f_start = max(f_hp - taper, 0)
            for i, f in enumerate(freqs):
                if f <= f_start:   mask[i] = 0.0
                elif f < f_hp:
                    mask[i] *= 0.5*(1 - np.cos(np.pi*(f-f_start)/max(f_hp-f_start,1e-10)))
        if self.chk_lp.isChecked():
            f_lp  = self.spin_lp.value()
            f_end = f_lp + taper
            for i, f in enumerate(freqs):
                if f >= f_end:   mask[i] = 0.0
                elif f > f_lp:
                    mask[i] *= 0.5*(1 + np.cos(np.pi*(f-f_lp)/max(taper,1e-10)))
        return mask

    def _preview(self):
        if self._data is None:
            QMessageBox.information(self,"No Data","Load a file first."); return
        if not self.chk_hp.isChecked() and not self.chk_lp.isChecked():
            QMessageBox.information(self,"No Filter","เปิด HP หรือ LP อย่างน้อย 1 อัน"); return

        N     = len(self._data)
        Y     = np.fft.rfft(self._data)
        freqs = np.fft.rfftfreq(N, d=1/self._fs)
        mask  = self._build_mask(freqs)
        amp_o = np.abs(Y)*2/N
        amp_f = np.abs(Y*mask)*2/N

        win = QWidget()
        win.setWindowTitle(f"Spectrum Preview — {self.comp_name}")
        win.setStyleSheet(STYLESHEET); win.resize(900, 420)
        plt.rcParams.update(MPLSTYLE)
        fig = Figure(figsize=(9, 4), tight_layout=True)
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        for ax in (ax1, ax2):
            ax.set_facecolor(DARK_BG)
            for sp in ax.spines.values(): sp.set_edgecolor(BORDER)

        mf = freqs > 0
        ax1.plot(freqs[mf], amp_o[mf], color=self.color, lw=1.0)
        ax1.fill_between(freqs[mf], amp_o[mf], alpha=0.15, color=self.color)
        if self.chk_hp.isChecked():
            ax1.axvline(self.spin_hp.value(), color=ACCENT3, lw=1.2, ls="--",
                        label=f"HP {self.spin_hp.value()} Hz")
        if self.chk_lp.isChecked():
            ax1.axvline(self.spin_lp.value(), color=ACCENT4, lw=1.2, ls="--",
                        label=f"LP {self.spin_lp.value()} Hz")
        ax1.set_xscale('log')
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:g}"))
        ax1.set_title(f"{self.comp_name} — Before Filter",  color=TEXT_PRI, pad=6)
        ax1.set_xlabel("Frequency (Hz)"); ax1.set_ylabel("Amplitude")
        ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER)

        ax2.plot(freqs[mf], amp_f[mf], color=ACCENT2, lw=1.0)
        ax2.fill_between(freqs[mf], amp_f[mf], alpha=0.15, color=ACCENT2)
        ax2.set_xscale('log')
        ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x,_: f"{x:g}"))
        ax2.set_title(f"{self.comp_name} — After Filter", color=ACCENT2, pad=6)
        ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Amplitude")

        canvas  = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, win)
        toolbar.setStyleSheet(f"background:{PANEL_BG};")

        btn_ok = QPushButton("✅  พอใจแล้ว → ไปขั้นตอน Convert Waveform")
        btn_ok.setObjectName("success"); btn_ok.setFixedHeight(38)
        btn_ok.clicked.connect(lambda: (win.close(),
                                        self.btn_ifft.setEnabled(True),
                                        self.lbl_status.setText("✓ Preview done — กด Convert")))
        vlay = QVBoxLayout(win)
        vlay.addWidget(toolbar); vlay.addWidget(canvas); vlay.addWidget(btn_ok)
        canvas.draw(); win.show()
        self._preview_win = win
        self._Y     = Y
        self._freqs = freqs
        self._mask  = mask

    def _run_ifft(self):
        if not hasattr(self, '_Y'):
            QMessageBox.information(self,"Run Preview first","กด Preview ก่อน"); return

        N = len(self._data)
        data_filt    = np.fft.irfft(self._Y * self._mask, n=N)
        self._modified = data_filt

        pga_o = np.max(np.abs(self._data))
        pga_f = np.max(np.abs(data_filt))

        parts = []
        if self.chk_hp.isChecked(): parts.append(f"HP>{self.spin_hp.value()}Hz")
        if self.chk_lp.isChecked(): parts.append(f"LP<{self.spin_lp.value()}Hz")
        filter_label = '  +  '.join(parts)

        # ── Window ────────────────────────────────────────────────────────
        win = QWidget()
        win.setWindowTitle(f"Modified Waveform — {self.comp_name}")
        win.setStyleSheet(STYLESHEET)
        win.resize(1100, 850)   # ขยายสูงขึ้นรองรับ 3 subplot

        plt.rcParams.update(MPLSTYLE)
        fig = Figure(figsize=(10, 9), tight_layout=True)

        ax1 = fig.add_subplot(3, 1, 1)   # Comparison
        ax2 = fig.add_subplot(3, 1, 2)   # Modified waveform
        ax3 = fig.add_subplot(3, 1, 3)   # Normalized ÷PGA

        for ax in (ax1, ax2, ax3):
            ax.set_facecolor(DARK_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(BORDER)

        # ── Subplot 1: Comparison ─────────────────────────────────────────
        ax1.plot(self._t, self._data,  color=self.color, lw=0.8,
                 alpha=0.7, label="Original")
        ax1.plot(self._t, data_filt,   color=ACCENT2,    lw=0.9,
                 label="Modified")
        ax1.set_title(f"{self.comp_name} — Comparison", color=TEXT_PRI, pad=6)
        ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Acceleration")
        ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER)

        # ── Subplot 2: Modified waveform ──────────────────────────────────
        ax2.plot(self._t, data_filt, color=ACCENT2, lw=0.8)
        ax2.fill_between(self._t, data_filt, alpha=0.1, color=ACCENT2)
        ax2.set_title(f"Modified  ({filter_label})", color=ACCENT2, pad=6)
        ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Acceleration")

        # ── Subplot 3: Normalized Acceleration ÷ PGA ─────────────────────
        norm = data_filt / pga_f if pga_f > 0 else data_filt

        ax3.plot(self._t, norm, color=ACCENT5, lw=0.8, alpha=0.9)
        ax3.fill_between(self._t, norm, 0,
                         where=(norm >= 0), alpha=0.12, color=ACCENT5)
        ax3.fill_between(self._t, norm, 0,
                         where=(norm < 0),  alpha=0.12, color=ACCENT3)

        # เส้น +1 และ -1
        ax3.axhline( 1.0, color=ACCENT5, lw=1.2, ls='--', alpha=0.9,
                     label=f"+1  (PGA = {pga_f:.5f})")
        ax3.axhline(-1.0, color=ACCENT3, lw=1.2, ls='--', alpha=0.9,
                     label=f"-1  (PGA = {pga_f:.5f})")
        ax3.axhline( 0.0, color=TEXT_SEC, lw=0.6, alpha=0.4)

        # ขีดสั้นๆ ที่ +1 และ -1 บนแกน Y (tick mark)
        ax3.set_ylim(-1.45, 1.45)
        ax3.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax3.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{x:+.1f}"))

        # หา Peak บนและล่าง แล้ววาด stem
        min_dist = max(int(self._fs * 0.3), 1)

        peaks_pos, _ = _fp( norm, height=0.5, distance=min_dist)
        peaks_neg, _ = _fp(-norm, height=0.5, distance=min_dist)

        # Peak บน (positive) — ขีดขึ้น
        for pk in peaks_pos:
            t_pk = self._t[pk]
            v_pk = norm[pk]
            ax3.plot([t_pk, t_pk], [v_pk, v_pk + 0.15],
                     color=ACCENT5, lw=1.4, alpha=0.85)
            ax3.plot(t_pk, v_pk, 'o', color=ACCENT5, ms=3.5, alpha=0.9)

        # Peak ล่าง (negative) — ขีดลง
        for pk in peaks_neg:
            t_pk = self._t[pk]
            v_pk = norm[pk]
            ax3.plot([t_pk, t_pk], [v_pk, v_pk - 0.15],
                     color=ACCENT3, lw=1.4, alpha=0.85)
            ax3.plot(t_pk, v_pk, 'o', color=ACCENT3, ms=3.5, alpha=0.9)

        ax3.set_title(
            f"{self.comp_name} — Normalized Acceleration  (÷ PGA = {pga_f:.5f})  "
            f"[{filter_label}]",
            color=ACCENT5, pad=6)
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("a / PGA")
        ax3.legend(fontsize=8, facecolor=CARD_BG, edgecolor=BORDER,
                   loc='upper right')

        # ── Footer stats ──────────────────────────────────────────────────
        fig.text(0.5, 0.002,
            f"PGA: {pga_o:.5f} → {pga_f:.5f}   |   "
            f"RMS: {np.sqrt(np.mean(self._data**2)):.5f} → "
            f"{np.sqrt(np.mean(data_filt**2)):.5f}",
            ha='center', color=TEXT_SEC, fontsize=9, fontfamily='monospace')

        canvas  = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, win)
        toolbar.setStyleSheet(f"background:{PANEL_BG};")

        vlay = QVBoxLayout(win)
        vlay.addWidget(toolbar)
        vlay.addWidget(canvas)
        canvas.draw()
        win.show()
        self._ifft_win = win

        self.lbl_status.setText(
            f"✓ Modified waveform ready  PGA {pga_f:.5f}")

    def get_modified(self):
        return self._modified if self._modified is not None else self._data


# ════════════════════════════════════════════════════════════════════════════
#  Tab 1 — FFT
# ════════════════════════════════════════════════════════════════════════════

class FFTTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw      = None
        self._dt       = 0.005
        self._acc_col  = 1
        self._has_time = False
        self._ncols    = 0
        self._mode     = "single"
        self._comp_data = {}
        self._t        = None
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(8)

        left = QVBoxLayout(); left.setSpacing(6)

        g_file = QGroupBox("📂  Load Signal File")
        vf = QVBoxLayout(g_file)
        btn_add = QPushButton("⬆  Load File")
        btn_add.clicked.connect(self._load_file)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.lbl_file.setWordWrap(True)
        for w in (btn_add, self.lbl_file): vf.addWidget(w)

        g_fft = QGroupBox("⚙  FFT Settings")
        v2 = QVBoxLayout(g_fft)
        self.cb_window = QComboBox()
        self.cb_window.addItems(["hann","hamming","blackman","rectangular"])
        self.cb_ymode = QComboBox()
        self.cb_ymode.addItems(["Amplitude (Linear)","Amplitude (dB)","Power Spectral Density"])
        self.chk_logx     = QCheckBox("Log X-axis"); self.chk_logx.setChecked(True)
        self.chk_onesided = QCheckBox("One-sided Spectrum"); self.chk_onesided.setChecked(True)
        self.chk_peaks    = QCheckBox("Show Peaks"); self.chk_peaks.setChecked(True)
        for w in (QLabel("Window:"), self.cb_window,
                  QLabel("Y-axis:"), self.cb_ymode,
                  self.chk_logx, self.chk_onesided, self.chk_peaks):
            v2.addWidget(w)

        g_smooth = QGroupBox("〰  Konno-Ohmachi Smoothing")
        v3 = QVBoxLayout(g_smooth)
        self.chk_smooth = QCheckBox("Enable Smoothing")
        lbl_b = QLabel("Bandwidth b:")
        self.spin_b = QDoubleSpinBox(); self.spin_b.setRange(5,200); self.spin_b.setValue(40)
        for w in (self.chk_smooth, lbl_b, self.spin_b): v3.addWidget(w)

        btn_run = QPushButton("⚡  Compute FFT")
        btn_run.setObjectName("success"); btn_run.clicked.connect(self.run)
        btn_save = QPushButton("💾  Save FFT Plot")
        btn_save.clicked.connect(self._save_fft)

        btn_export = QPushButton("💾  Export All Modified Waveforms (.csv)")
        btn_export.setObjectName("purple")
        btn_export.clicked.connect(self._export_modified)

        g_stat = QGroupBox("📊  Signal Info")
        vs = QVBoxLayout(g_stat)
        self.stat_box = QTextEdit(); self.stat_box.setReadOnly(True)
        self.stat_box.setFixedHeight(160)
        vs.addWidget(self.stat_box)

        for w in (g_file, g_fft, g_smooth, btn_run, btn_save, btn_export, g_stat):
            left.addWidget(w)
        left.addStretch()

        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(275)
        scroll = QScrollArea(); scroll.setWidget(lw)
        scroll.setWidgetResizable(True); scroll.setFixedWidth(285)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setSpacing(6)

        right_scroll = QScrollArea()
        right_scroll.setWidget(self.right_widget)
        right_scroll.setWidgetResizable(True)

        root.addWidget(scroll)
        root.addWidget(right_scroll, stretch=1)

        self._placeholder = QLabel("  ← Load a file to begin")
        self._placeholder.setStyleSheet(f"color:{TEXT_SEC}; font-size:14px;")
        self.right_layout.addWidget(self._placeholder)
        self.right_layout.addStretch()

        self._filter_panels = {}

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Signal File", "",
            "Data Files (*.csv *.txt *.dat *.out);;All Files (*)")
        if not path: return

        try:
            raw = _smart_genfromtxt(path)
            if raw.ndim == 1:
                raw = raw[~np.isnan(raw)].reshape(-1, 1)
            else:
                raw = raw[~np.isnan(raw).any(axis=1)]
            if raw.shape[0] == 0:
                QMessageBox.warning(self, "Empty File",
                    "ไม่พบข้อมูลตัวเลขในไฟล์"); return
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e)); return

        self._raw   = raw
        self._ncols = raw.shape[1]

        dlg = FileImportDialog(os.path.basename(path), self._ncols, self)
        dlg.confirmed.connect(self._on_import_confirmed)
        dlg.show()

    def _on_import_confirmed(self, dt, acc_col):
        self._dt      = dt
        self._acc_col = acc_col
        raw   = self._raw
        ncols = self._ncols
        fs    = 1.0 / dt

        remaining_cols = ncols - acc_col
        if remaining_cols >= 3:
            self._mode     = "3comp"
            self._has_time = acc_col > 0
            ew   = raw[:, acc_col    ].astype(float)
            ns   = raw[:, acc_col + 1].astype(float)
            v    = raw[:, acc_col + 2].astype(float)
            N    = len(ew)
            self._t = np.arange(N) * dt
            self._comp_data = {"EW": ew, "NS": ns, "V": v}
        else:
            self._mode     = "single"
            self._has_time = acc_col > 0
            acc  = raw[:, acc_col].astype(float)
            N    = len(acc)
            self._t = np.arange(N) * dt
            self._comp_data = {"ACC": acc}

        self._fs = fs

        dur = N * dt
        info = (f"{ncols} cols | {N} samples | dt={dt:.5f}s | "
                f"fs={fs:.2f}Hz | dur={dur:.2f}s\n"
                f"Mode: {'3-Component (EW/NS/V)' if self._mode=='3comp' else 'Single (ACC)'}")
        self.lbl_file.setText(info)
        self.stat_box.setText(info)

        self._rebuild_right_panel(fs, dt)

    def _rebuild_right_panel(self, fs, dt):
        for i in reversed(range(self.right_layout.count())):
            item = self.right_layout.itemAt(i)
            if item.widget(): item.widget().deleteLater()

        self._filter_panels = {}
        comp_colors = {"EW": ACCENT, "NS": ACCENT2, "V": ACCENT4, "ACC": ACCENT}

        for comp, data in self._comp_data.items():
            color = comp_colors.get(comp, ACCENT)

            time_canvas = PlotCanvas(1, 1, figsize=(9, 2))
            ax = time_canvas.axes[0]
            ax.plot(self._t, data, color=color, lw=0.7, alpha=0.9)
            ax.set_title(f"Component: {comp}", color=color, pad=5)
            ax.set_xlabel("Time (s)"); ax.set_ylabel("Acc")
            time_canvas.draw()
            time_canvas.setFixedHeight(210)
            self.right_layout.addWidget(time_canvas)

            fft_canvas = PlotCanvas(1, 1, figsize=(9, 2))
            fft_canvas.axes[0].set_title(f"FFT — {comp}  (press Compute FFT)",
                                          color=TEXT_SEC, pad=5)
            fft_canvas.draw()
            fft_canvas.setFixedHeight(210)
            self.right_layout.addWidget(fft_canvas)
            setattr(self, f"_fft_canvas_{comp}", fft_canvas)

            panel = ComponentFilterPanel(comp, color)
            panel.set_data(data, fs, self._t)
            self._filter_panels[comp] = panel
            self.right_layout.addWidget(panel)

            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color:{BORDER}; margin:6px 0;")
            self.right_layout.addWidget(sep)

        self.right_layout.addStretch()

    def run(self):
        if not self._comp_data:
            QMessageBox.information(self,"No Data","Load a file first."); return

        window  = self.cb_window.currentText()
        ymode   = self.cb_ymode.currentText()
        logx    = self.chk_logx.isChecked()
        one     = self.chk_onesided.isChecked()
        smooth  = self.chk_smooth.isChecked()
        b_val   = self.spin_b.value()
        peaks   = self.chk_peaks.isChecked()
        fs      = self._fs

        ylabel_map = {
            "Amplitude (Linear)":    "Amplitude",
            "Amplitude (dB)":        "Magnitude (dB)",
            "Power Spectral Density":"PSD (units²/Hz)",
        }
        ylabel = ylabel_map[ymode]

        stats = []
        comp_colors = {"EW": ACCENT, "NS": ACCENT2, "V": ACCENT4, "ACC": ACCENT}

        for comp, data in self._comp_data.items():
            color = comp_colors.get(comp, ACCENT)
            N = len(data)
            freqs, amp = compute_fft(data, fs, window=window,
                                     ymode=ymode, one_sided=one)
            if smooth and len(freqs) > 0:
                msk = freqs > 0
                amp_s = amp.copy()
                amp_s[msk] = konno_ohmachi_smooth(freqs[msk], amp[msk], b=b_val)
                amp = amp_s

            canvas_attr = f"_fft_canvas_{comp}"
            if hasattr(self, canvas_attr):
                cv = getattr(self, canvas_attr)
                cv.clear_all()
                ax = cv.axes[0]
                mf = freqs > 0
                ax.plot(freqs[mf], amp[mf], color=color, lw=1.0)
                ax.fill_between(freqs[mf], amp[mf], alpha=0.12, color=color)

                if peaks:
                    pf, pa = find_peaks_fft(freqs[mf], amp[mf], n=3)
                    for f_, a_ in zip(pf, pa):
                        ax.axvline(f_, color=color, alpha=0.5, lw=0.8, ls="--")
                        ax.annotate(f"{f_:.3f}Hz", xy=(f_,a_),
                            xytext=(4,4), textcoords="offset points",
                            color=color, fontsize=7,
                            bbox=dict(boxstyle="round,pad=0.2",
                                      fc=CARD_BG, ec=BORDER, alpha=0.8))
                if logx:
                    ax.set_xscale('log')
                    ax.xaxis.set_major_formatter(
                        ticker.FuncFormatter(lambda x,_: f"{x:g}"))

                title = f"FFT — {comp}"
                if smooth: title += f"  |  KO b={b_val}"
                ax.set_title(title, color=color, pad=5)
                ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel(ylabel)
                cv.draw()

            pf, pa = find_peaks_fft(freqs[freqs>0], amp[freqs>0]) if len(freqs)>0 else ([],[])
            stats.append(f"── {comp} ──")
            stats.append(f"  N={N}  Δf={fs/N:.5f}Hz  fs={fs:.2f}Hz")
            for f_,a_ in zip(pf[:3], pa[:3]):
                stats.append(f"  Peak {f_:.3f}Hz  ({a_:.4f})")

        self.stat_box.setText("\n".join(stats))

    def _export_modified(self):
        if not self._filter_panels:
            QMessageBox.information(self,"No Data","Load a file first."); return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Modified Waveforms", "modified_waveform.csv",
            "CSV (*.csv)")
        if not path: return

        cols  = []
        names = []
        for comp, panel in self._filter_panels.items():
            d = panel.get_modified()
            if d is None:
                QMessageBox.warning(self,"Missing Data",
                    f"Component {comp} ยังไม่ได้ Convert\n"
                    "จะใช้ original signal แทน"); d = self._comp_data[comp]
            cols.append(d); names.append(comp)

        N_min = min(len(c) for c in cols)
        out   = np.column_stack([self._t[:N_min]] + [c[:N_min] for c in cols])
        header = "Time_s," + ",".join(names)
        np.savetxt(path, out, delimiter=",", header=header, comments="")
        QMessageBox.information(self,"Saved",
            f"Exported {len(names)} component(s):\n{path}\n"
            f"Columns: {header}")

    def _save_fft(self):
        if not self._comp_data:
            QMessageBox.information(self,"No Data","Load and compute FFT first."); return
        path, _ = QFileDialog.getSaveFileName(
            self,"Save","fft.png","PNG (*.png);;PDF (*.pdf)")
        if path:
            comp = list(self._comp_data.keys())[0]
            cv = getattr(self, f"_fft_canvas_{comp}", None)
            if cv: cv.fig.savefig(path, dpi=150, bbox_inches="tight")


# ════════════════════════════════════════════════════════════════════════════
#  Tab 2 — Acc / Vel / Disp
# ════════════════════════════════════════════════════════════════════════════

class AccVelDispTab(QWidget):
    COMP_COLORS = {"EW": ACCENT, "NS": ACCENT2, "V": ACCENT4, "ACC": ACCENT}

    def __init__(self, parent=None):
        super().__init__(parent)
        self._raw   = None
        self._t     = None
        self._fs    = None
        self._comp_data  = {}
        self._comp_results = {}
        self._build()

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6); root.setSpacing(8)

        left = QVBoxLayout(); left.setSpacing(6)

        g_file = QGroupBox("📂  Load Signal File")
        vf = QVBoxLayout(g_file)
        btn_load = QPushButton("⬆  Load File")
        btn_load.clicked.connect(self._load)
        self.lbl_file = QLabel("No file loaded")
        self.lbl_file.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")
        self.lbl_file.setWordWrap(True)

        lbl_fs = QLabel("Sampling Rate (Hz):")
        self.spin_fs = QDoubleSpinBox()
        self.spin_fs.setRange(0.01, 1e6); self.spin_fs.setValue(100)
        self.spin_fs.setSuffix(" Hz")

        lbl_col = QLabel("Acc start column (0-based):")
        self.spin_col = QSpinBox(); self.spin_col.setRange(0, 99)

        for w in (btn_load, self.lbl_file, lbl_fs, self.spin_fs,
                  lbl_col, self.spin_col):
            vf.addWidget(w)

        g_sel = QGroupBox("📌  Select Component to Convert")
        vs = QVBoxLayout(g_sel)
        lbl_comp = QLabel("Compute for axis:")
        self.cb_comp = QComboBox()
        self.cb_comp.addItems(["All components", "EW only", "NS only", "V only", "ACC only"])
        for w in (lbl_comp, self.cb_comp): vs.addWidget(w)

        g_type = QGroupBox("📌  Input Signal Type")
        vt = QVBoxLayout(g_type)
        self.cb_input = QComboBox()
        self.cb_input.addItems([
            "Acceleration  (m/s²)",
            "Velocity      (m/s)",
            "Displacement  (m)",
        ])
        for w in (QLabel("Input type:"), self.cb_input): vt.addWidget(w)

        g_unit = QGroupBox("⚖  Unit Conversion")
        vu = QVBoxLayout(g_unit)
        lbl_factor = QLabel("Multiply input by factor before compute:")
        lbl_factor.setStyleSheet(f"color:{TEXT_SEC}; font-size:10px;")

        self.cb_unit = QComboBox()
        self.cb_unit.addItems([
            "1.0         (no conversion)",
            "9.81        (g  ->  m/s2)",
            "0.01        (cm/s2  ->  m/s2)",
            "0.001       (mm/s2  ->  m/s2)",
            "Custom...",
        ])
        self.cb_unit.currentIndexChanged.connect(self._on_unit_changed)

        self.spin_factor = QDoubleSpinBox()
        self.spin_factor.setRange(-1e9,1e9)
        self.spin_factor.setDecimals(4)
        self.spin_factor.setValue(9.81)
        self.spin_factor.setEnabled(False)
        self.cb_unit.setCurrentIndex(1)

        for w in (lbl_factor, self.cb_unit, self.spin_factor):
            vu.addWidget(w)

        g_base = QGroupBox("⚙  Pre-processing")
        vb = QVBoxLayout(g_base)
        self.chk_detrend = QCheckBox("Detrend (remove linear trend)")
        self.chk_detrend.setChecked(True)
        self.chk_demean = QCheckBox("Demean (remove mean)")
        self.chk_demean.setChecked(True)
        for w in (self.chk_detrend, self.chk_demean): vb.addWidget(w)

        btn_run = QPushButton("⚡  Convert")
        btn_run.setObjectName("success")
        btn_run.clicked.connect(self.run)

        g_exp = QGroupBox("💾  Export")
        ve = QVBoxLayout(g_exp)
        self.cb_exp_type = QComboBox()
        self.cb_exp_type.addItems(["Acceleration", "Velocity", "Displacement"])
        btn_save_sel = QPushButton("Save Selected Type (all axes)")
        btn_save_sel.clicked.connect(self._save_selected_type)
        btn_save_all = QPushButton("Save All (Acc + Vel + Disp × axes)")
        btn_save_all.setObjectName("purple")
        btn_save_all.clicked.connect(self._save_all)
        for w in (QLabel("Export type:"), self.cb_exp_type,
                  btn_save_sel, btn_save_all):
            ve.addWidget(w)

        g_stat = QGroupBox("📊  Stats")
        vst = QVBoxLayout(g_stat)
        self.stat_box = QTextEdit()
        self.stat_box.setReadOnly(True); self.stat_box.setFixedHeight(200)
        vst.addWidget(self.stat_box)

        for w in (g_file, g_sel, g_type, g_unit, g_base, btn_run, g_exp, g_stat):
            left.addWidget(w)
        left.addStretch()

        lw = QWidget(); lw.setLayout(left); lw.setFixedWidth(280)
        scroll = QScrollArea(); scroll.setWidget(lw)
        scroll.setWidgetResizable(True); scroll.setFixedWidth(292)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.right_widget = QWidget()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setSpacing(4)
        self.right_layout.addWidget(
            QLabel("  <- Load a file and click Convert"), 0)
        self.right_layout.addStretch()

        right_scroll = QScrollArea()
        right_scroll.setWidget(self.right_widget)
        right_scroll.setWidgetResizable(True)

        root.addWidget(scroll)
        root.addWidget(right_scroll, stretch=1)

    def _on_unit_changed(self, idx):
        factor_map = {0: 1.0, 1: 9.81, 2: 0.01, 3: 0.001}
        if idx in factor_map:
            self.spin_factor.setValue(factor_map[idx])
            self.spin_factor.setEnabled(False)
        else:
            self.spin_factor.setEnabled(True)
            self.spin_factor.setFocus()

    def _load(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Signal File", "",
            "Data Files (*.csv *.txt *.dat *.out);;All Files (*)")
        if not path: return
        try:
            raw = _smart_genfromtxt(path)
            self._raw = raw
            ncols = raw.shape[1]
            self.cb_comp.clear()
            if ncols >= 3:
                self.cb_comp.addItems(["All components",
                                       "EW only", "NS only", "V only"])
            else:
                self.cb_comp.addItems(["ACC only"])
            self.lbl_file.setText(
                f"{os.path.basename(path)}\n{raw.shape[0]} samples, {ncols} col(s)")
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def run(self):
        if self._raw is None:
            QMessageBox.information(self, "No Data", "Load a file first.")
            return

        raw     = self._raw
        fs      = self.spin_fs.value()
        dt      = 1.0 / fs
        acc_col = min(self.spin_col.value(), raw.shape[1] - 1)
        ncols   = raw.shape[1]
        N       = raw.shape[0]
        t       = np.arange(N) * dt
        self._t = t; self._fs = fs

        remaining = ncols - acc_col
        if remaining >= 3:
            self._comp_data = {
                "EW": raw[:, acc_col    ].astype(float),
                "NS": raw[:, acc_col + 1].astype(float),
                "V":  raw[:, acc_col + 2].astype(float),
            }
        else:
            self._comp_data = {"ACC": raw[:, acc_col].astype(float)}

        sel = self.cb_comp.currentText()
        if sel == "All components":
            comps_to_run = self._comp_data
        else:
            key = sel.replace(" only", "")
            comps_to_run = {key: self._comp_data[key]} if key in self._comp_data else self._comp_data

        idx = self.cb_input.currentIndex()
        self._comp_results = {}
        stats_lines = []

        for comp, sig in comps_to_run.items():
            s = sig.copy()
            factor = self.spin_factor.value()
            if factor != 1.0:
                s = s * factor
            if self.chk_demean.isChecked():  s = s - np.mean(s)
            if self.chk_detrend.isChecked(): s = scipy_signal.detrend(s)

            if idx == 0:
                acc  = s
                vel  = cumulative_trapezoid(acc, dx=dt, initial=0)
                disp = cumulative_trapezoid(vel, dx=dt, initial=0)
            elif idx == 1:
                vel  = s
                disp = cumulative_trapezoid(vel, dx=dt, initial=0)
                acc  = np.gradient(vel, dt)
            else:
                disp = s
                vel  = np.gradient(disp, dt)
                acc  = np.gradient(vel, dt)

            self._comp_results[comp] = {"acc": acc, "vel": vel, "disp": disp}
            stats_lines += [
                f"-- {comp} --",
                f"  PGA : {np.max(np.abs(acc)):.6f} m/s2",
                f"  PGV : {np.max(np.abs(vel)):.6f} m/s",
                f"  PGD : {np.max(np.abs(disp)):.6f} m",
            ]

        self.stat_box.setText(
            f"N={N}  dur={N/fs:.3f}s  fs={fs:.1f}Hz\n" +
            "\n".join(stats_lines))

        self._rebuild_plots(t)

    def _rebuild_plots(self, t):
        for i in reversed(range(self.right_layout.count())):
            item = self.right_layout.itemAt(i)
            if item.widget(): item.widget().deleteLater()

        comp_colors = self.COMP_COLORS
        signal_types = [
            ("Acceleration (m/s²)", "acc", ACCENT),
            ("Velocity (m/s)",      "vel", ACCENT2),
            ("Displacement (m)",    "disp", ACCENT4),
        ]

        for comp, results in self._comp_results.items():
            color = comp_colors.get(comp, ACCENT)

            lbl = QLabel(f"  Component: {comp}")
            lbl.setStyleSheet(f"color:{color}; font-size:13px; font-weight:bold; padding:4px 0;")
            self.right_layout.addWidget(lbl)

            canvas = PlotCanvas(3, 1, figsize=(9, 6))
            for ax, (title, key, c) in zip(canvas.axes, signal_types):
                ax.plot(t, results[key], color=c, lw=0.8)
                ax.set_title(f"{comp} — {title}", color=c, pad=5)
                ax.set_xlabel("Time (s)")
            canvas.draw()
            canvas.setFixedHeight(460)
            self.right_layout.addWidget(canvas)

            sep = QFrame(); sep.setFrameShape(QFrame.HLine)
            sep.setStyleSheet(f"color:{BORDER}; margin:4px 0;")
            self.right_layout.addWidget(sep)

        self.right_layout.addStretch()

    def _save_selected_type(self):
        if not self._comp_results:
            QMessageBox.information(self, "No Data", "Click Convert first."); return
        sig_key = self.cb_exp_type.currentText().lower()
        key_map = {"acceleration": "acc", "velocity": "vel", "displacement": "disp"}
        k = key_map[sig_key]

        path, _ = QFileDialog.getSaveFileName(
            self, f"Save {sig_key}", f"{sig_key}_all_components.csv", "CSV (*.csv)")
        if not path: return

        cols  = [self._t]
        names = ["Time_s"]
        for comp, res in self._comp_results.items():
            cols.append(res[k]); names.append(f"{comp}_{k}")

        N = min(len(c) for c in cols)
        np.savetxt(path, np.column_stack([c[:N] for c in cols]),
                   delimiter=",", header=",".join(names), comments="")
        QMessageBox.information(self, "Saved", f"Saved: {path}")

    def _save_all(self):
        if not self._comp_results:
            QMessageBox.information(self, "No Data", "Click Convert first."); return
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not folder: return

        key_map = {"acc": "acceleration", "vel": "velocity", "disp": "displacement"}
        saved = []
        for sig_k, sig_name in key_map.items():
            cols  = [self._t]
            names = ["Time_s"]
            for comp, res in self._comp_results.items():
                cols.append(res[sig_k]); names.append(f"{comp}_{sig_k}")
            N = min(len(c) for c in cols)
            fpath = os.path.join(folder, f"{sig_name}.csv")
            np.savetxt(fpath, np.column_stack([c[:N] for c in cols]),
                       delimiter=",", header=",".join(names), comments="")
            saved.append(fpath)

        QMessageBox.information(self, "Saved",
            f"Saved {len(saved)} files:\n" + "\n".join(saved))


# ════════════════════════════════════════════════════════════════════════════
#  Main Window
# ════════════════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Seismic Signal Analyzer  —  Patrick\'s Earthquake Engineering Lab")
        self.resize(1300, 820)
        self.setStyleSheet(STYLESHEET)

        tabs = QTabWidget()
        tabs.addTab(FFTTab(),        "📊  FFT + Filter")
        tabs.addTab(AccVelDispTab(), "📐  Acc / Vel / Disp")

        self.setCentralWidget(tabs)
        sb = QStatusBar(); self.setStatusBar(sb)
        sb.showMessage("Seismic Signal Analyzer  |  Patrick\'s Earthquake Engineering Lab  |  Ready")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()