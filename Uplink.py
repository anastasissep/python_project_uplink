import sys
import math
import csv
import os
import time  # ✅ FIX 1: needed for time.time() in live metrics
from datetime import datetime
import urllib.request
from collections import deque

import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QHBoxLayout, QVBoxLayout, QGridLayout,
    QLabel, QComboBox, QDoubleSpinBox, QPlainTextEdit,
    QCheckBox, QListWidgetItem,
    QPushButton, QGroupBox, QStackedWidget,
    QProgressBar, QFrame, QDockWidget,
    QTableWidget, QTableWidgetItem, QFileDialog,
    QHeaderView, QMessageBox, QTabWidget,
    QSizePolicy, QButtonGroup,
    QScrollArea, QLayout  # ✅ NEW
)
from PyQt5.QtCore import Qt, QTimer
import pyqtgraph as pg

# Optional: satellite TLE (Skyfield)
try:
    from skyfield.api import EarthSatellite, load, wgs84
    SKYFIELD_OK = True
except Exception:
    SKYFIELD_OK = False


# ================== HELPERS / MODELS ==================

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fspl_db(f_mhz, d_km):
    d_km = max(d_km, 1e-9)
    return 32.45 + 20.0 * math.log10(f_mhz) + 20.0 * math.log10(d_km)


def dish_gain_dbi(d_m, f_mhz, eff):
    c = 299792458.0
    f_hz = f_mhz * 1e6
    lam = c / max(f_hz, 1.0)
    eff = clamp(eff, 0.05, 0.95)
    g_lin = eff * (math.pi * d_m / max(lam, 1e-12)) ** 2
    return 10.0 * math.log10(max(g_lin, 1e-12))


def hpbw_deg(d_m, f_mhz):
    c = 299792458.0
    f_hz = f_mhz * 1e6
    lam = c / max(f_hz, 1.0)
    return 70.0 * (lam / max(d_m, 1e-6))


def pointing_loss_db(mispoint_deg, hpbw_deg_val):
    ratio = (mispoint_deg / max(hpbw_deg_val, 1e-6))
    return 12.0 * (ratio ** 2)


def noise_power_dbm(bw_hz, nf_db):
    return -174.0 + 10.0 * math.log10(max(bw_hz, 1.0)) + nf_db


# --- Conversions / Satellite-theory helpers ---

K_BOLTZ_DBW_HZ_K = -228.6  # 10log10(k) in dBW/Hz/K


def dbm_to_dbw(p_dbm: float) -> float:
    return float(p_dbm) - 30.0


def dbw_to_dbm(p_dbw: float) -> float:
    return float(p_dbw) + 30.0


def sat_gain_from_hpbw_dbi(theta_3db_deg: float, eff: float) -> float:
    """
    Satellite receive antenna gain from 3 dB beamwidth (deg), using the same
    parabolic approximation used elsewhere:
        HPBW(deg) ≈ 70 * (λ / D)
        G(dBi) ≈ 10log10(η * (πD/λ)^2)
    Eliminating D/λ gives:
        G(dBi) ≈ 10log10(η) + 20log10(π*70/HPBW)

    This matches the theory example 2.5.6/2.5.7 when combined with Tsys model below.
    """
    theta = max(float(theta_3db_deg), 1e-6)
    eff = clamp(float(eff), 0.05, 0.95)
    return 10.0 * math.log10(eff) + 20.0 * math.log10((math.pi * 70.0) / theta)


def rx_system_noise_temp_k(ta_k: float, tf_k: float, lfrx_db: float, frx_db: float) -> float:
    """
    System noise temperature referred to antenna input, for a feeder loss Lfrx before LNA:
        Trx = (F-1)*290
        Tsys = Ta + (L-1)*Tf + L*Trx
    where L and F are linear.
    """
    ta_k = max(float(ta_k), 1.0)
    tf_k = max(float(tf_k), 1.0)
    L = 10.0 ** (float(lfrx_db) / 10.0)
    F = 10.0 ** (float(frx_db) / 10.0)
    Trx = max((F - 1.0) * 290.0, 0.0)
    Tsys = ta_k + (L - 1.0) * tf_k + L * Trx
    return max(Tsys, 1.0)


def make_spin(sb: QDoubleSpinBox, lo, hi, val, decimals=2, step=1.0, suffix="", tooltip=""):
    sb.setRange(lo, hi)
    sb.setDecimals(decimals)
    sb.setValue(val)
    sb.setSingleStep(step)
    sb.setSuffix(suffix)
    sb.setKeyboardTracking(False)
    sb.setAccelerated(True)
    if tooltip:
        sb.setToolTip(tooltip)
    return sb


def make_ro_box(decimals=4, suffix=""):
    sb = QDoubleSpinBox()
    sb.setRange(-1e12, 1e12)
    sb.setDecimals(decimals)
    sb.setKeyboardTracking(False)
    sb.setButtonSymbols(QDoubleSpinBox.NoButtons)
    sb.setReadOnly(True)
    sb.setSuffix(suffix)
    return sb


# ✅ NEW helper (one time)
def wrap_in_scroll(widget: QWidget) -> QScrollArea:
    sa = QScrollArea()
    sa.setWidget(widget)
    sa.setWidgetResizable(True)
    sa.setFrameShape(QFrame.NoFrame)
    sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    return sa


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def safe_read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_tle_text(tle_text: str):
    """
    Accepts:
    - 3-line blocks: NAME, 1 ..., 2 ...
    - 2-line blocks: 1 ..., 2 ... (name becomes SAT-<n>)
    Returns list of (name, line1, line2)
    """
    lines = [ln.strip() for ln in tle_text.splitlines() if ln.strip()]
    out = []
    i = 0
    n2 = 0
    while i < len(lines):
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            n2 += 1
            name = f"SAT-{n2}"
            l1 = lines[i]
            l2 = lines[i + 1]
            out.append((name, l1, l2))
            i += 2
        elif (i + 2 < len(lines) and
              (not lines[i].startswith("1 ")) and
              lines[i + 1].startswith("1 ") and
              lines[i + 2].startswith("2 ")):
            name = lines[i]
            l1 = lines[i + 1]
            l2 = lines[i + 2]
            out.append((name, l1, l2))
            i += 3
        else:
            i += 1
    return out


class SatelliteTracker:
    """
    Optional real satellite tracking using TLE (Skyfield).
    - Can load many satellites (TLE set)
    - Select active satellite
    Safe: if skyfield not available, stays disabled.
    """
    def __init__(self):
        self.enabled = False
        self.loaded = False

        self._ts = None
        self._sats = []          # list[EarthSatellite]
        self._names = []         # list[str]
        self._active_idx = 0

        self.last_lat = None
        self.last_lon = None
        self.last_alt_km = None
        self.last_slant_km = None

    def set_enabled(self, en: bool):
        self.enabled = bool(en)

    def names(self):
        return list(self._names)

    def set_active_index(self, idx: int):
        if not self._sats:
            self._active_idx = 0
            return
        self._active_idx = int(clamp(idx, 0, len(self._sats) - 1))

    def active_name(self):
        if not self._names:
            return ""
        return self._names[self._active_idx]

    def load_tle_list(self, tle_list):
        """
        tle_list: list of (name, l1, l2)
        """
        if not SKYFIELD_OK:
            self.loaded = False
            self._sats = []
            self._names = []
            self._active_idx = 0
            return False, "Skyfield not installed"

        try:
            self._ts = load.timescale()
            sats = []
            names = []
            for (name, l1, l2) in tle_list:
                try:
                    sat = EarthSatellite(l1.strip(), l2.strip(), name.strip() or "SAT", self._ts)
                    sats.append(sat)
                    names.append(sat.name)
                except Exception:
                    continue

            self._sats = sats
            self._names = names
            self._active_idx = 0
            self.loaded = len(self._sats) > 0
            if not self.loaded:
                return False, "No valid satellites found in TLE."
            return True, f"Loaded {len(self._sats)} satellites"
        except Exception as e:
            self.loaded = False
            self._sats = []
            self._names = []
            self._active_idx = 0
            return False, str(e)

    def update(self, ground_lat_deg: float, ground_lon_deg: float, when_utc=None):
        """
        Updates active satellite geodetic + slant range to ground station.
        Returns: (ok, lat_deg, lon_deg, alt_km, slant_km)
        """
        if not (SKYFIELD_OK and self.enabled and self.loaded and self._sats):
            return False, None, None, None, None

        try:
            if when_utc is None:
                when_utc = datetime.utcnow()

            t = self._ts.utc(
                when_utc.year, when_utc.month, when_utc.day,
                when_utc.hour, when_utc.minute, when_utc.second + when_utc.microsecond * 1e-6
            )

            sat = self._sats[self._active_idx]

            geoc = sat.at(t)
            sp = wgs84.subpoint(geoc)
            lat = sp.latitude.degrees
            lon = sp.longitude.degrees
            alt_km = sp.elevation.km

            gs = wgs84.latlon(ground_lat_deg, ground_lon_deg)
            topo = (sat - gs).at(t)
            slant_km = topo.distance().km

            self.last_lat = lat
            self.last_lon = lon
            self.last_alt_km = alt_km
            self.last_slant_km = slant_km

            return True, lat, lon, alt_km, slant_km
        except Exception:
            return False, None, None, None, None


class NavButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(44)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


# ================== MAIN APP ==================

class UplinkApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Uplink Antenna Simulator – Professional Edition (Full)")
        self.resize(1550, 880)

        self.history_rows = []

        # Optional real satellite tracking (TLE)
        self.sat_tracker = SatelliteTracker()

        # Offline map background item
        self._map_bg_item = None

        # 2.5D link-view trail (sat path)
        self._sat_trail = deque(maxlen=240)  # ~8s at 30fps

        # Where to cache online-downloaded TLEs
        self.tle_cache_dir = os.path.join(os.getcwd(), "tle_cache")
        ensure_dir(self.tle_cache_dir)

        # Live metrics buffers (for new "Live Metrics" tab)
        self._live_n = 360
        self._live_x = deque(maxlen=self._live_n)
        self._live_pr = deque(maxlen=self._live_n)
        self._live_snr = deque(maxlen=self._live_n)
        self._live_ebn0 = deque(maxlen=self._live_n)
        self._live_margin = deque(maxlen=self._live_n)
        self._live_t0 = None

        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(12, 12, 12, 12)
        main.setSpacing(12)

        # -------- Navigation (left, boxed) --------
        self.nav_panel = QWidget()
        nav_layout = QVBoxLayout(self.nav_panel)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(10)

        nav_title = QLabel("Navigation")
        nav_title.setObjectName("NavTitle")
        nav_layout.addWidget(nav_title)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.nav_buttons = []
        self.nav_names = [
            "Uplink Chain",
            "Environment",
            "Link Budget",
            "Visualization",
            "Scenarios",
            "Advanced Settings",
            "Formulas / Theory",
            "References"
        ]
        for i, name in enumerate(self.nav_names):
            b = NavButton(name)
            b.setObjectName("NavBtn")
            self.nav_group.addButton(b, i)
            self.nav_buttons.append(b)
            nav_layout.addWidget(b)

        nav_layout.addStretch(1)

        self.nav_panel.setFixedWidth(260)

        # -------- Pages (center) --------
        self.stack = QStackedWidget()

        main.addWidget(self.nav_panel)
        main.addWidget(self.stack, 1)

        # Build pages
        self.build_uplink_page()
        self.build_environment_page()
        self.build_budget_page()
        self.build_visualization_page()
        self.build_scenarios_page()
        self.build_advanced_page()
        self.build_formulas_page()
        self.build_references_page()

        # Right dock (always visible)
        self.build_global_results_dock()

        # Signals
        self.connect_all_signals()

        # default selection
        self.nav_buttons[0].setChecked(True)
        self.stack.setCurrentIndex(0)

        # ---- Animation (for Link View fancy upgrades) ----
        self._anim_phase = 0.0
        self._last_link_view = None
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._tick_animation)
        self._anim_timer.start(33)  # ~30 FPS

        # Init defaults and compute
        self.reset_defaults()
        self.update_results()

    # ================== Animation tick ==================

    def _tick_animation(self):
        self._anim_phase += 0.06
        if self._anim_phase > 2 * math.pi:
            self._anim_phase -= 2 * math.pi

        if self._last_link_view is not None:
            color, hpbw, snr = self._last_link_view
            self.update_link_view(color, hpbw, snr, animated=True)

    # ================== UI PAGES ==================

    def build_uplink_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        box_tx = QGroupBox("Transmitter (UE)")
        g = QGridLayout(box_tx)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

        self.cb_mod = QComboBox()
        self.cb_mod.addItems(["QPSK", "16QAM", "64QAM", "256QAM"])
        self.cb_mod.setToolTip("Modulation selection (affects required Eb/N0).")

        self.sb_bw_mhz = make_spin(
            QDoubleSpinBox(),
            0.001, 50000.0, 20.0,
            decimals=3, step=5.0, suffix=" MHz",
            tooltip="Bandwidth B (MHz)\nNoise(dBm) = -174 + 10log10(B_Hz) + NF"
        )

        self.sb_tx_pwr_dbm = make_spin(
            QDoubleSpinBox(),
            -120.0, 80.0, 23.0,
            decimals=2, step=1.0, suffix=" dBm",
            tooltip="Transmit Power Ptx (dBm)"
        )

        self.sb_rb_mbps = make_spin(
            QDoubleSpinBox(),
            0.001, 200000.0, 10.0,
            decimals=3, step=1.0, suffix=" Mbps",
            tooltip="Bit Rate Rb (Mbps)\nEb/N0 = C/N0 - 10log10(Rb_Hz)"
        )

        g.addWidget(QLabel("Modulation:"), 0, 0)
        g.addWidget(self.cb_mod, 0, 1)
        g.addWidget(QLabel("Bandwidth B:"), 1, 0)
        g.addWidget(self.sb_bw_mhz, 1, 1)
        g.addWidget(QLabel("Tx Power Ptx:"), 2, 0)
        g.addWidget(self.sb_tx_pwr_dbm, 2, 1)
        g.addWidget(QLabel("Bit Rate Rb:"), 3, 0)
        g.addWidget(self.sb_rb_mbps, 3, 1)

        box_ant = QGroupBox("Antennas (Basic + Beamwidth / Pointing)")
        g2 = QGridLayout(box_ant)
        g2.setHorizontalSpacing(12)
        g2.setVerticalSpacing(10)

        self.sb_ue_gain = make_spin(QDoubleSpinBox(), -30.0, 90.0, 2.0, decimals=2, step=0.5, suffix=" dBi")
        self.sb_gnb_gain = make_spin(QDoubleSpinBox(), -30.0, 90.0, 18.0, decimals=2, step=0.5, suffix=" dBi")
        self.sb_cable_loss = make_spin(
            QDoubleSpinBox(), 0.0, 200.0, 0.0, decimals=2, step=0.5, suffix=" dB",
            tooltip="Tx losses Ltx (dB)\nEIRP = Ptx + Gtx - Ltx"
        )

        self.sb_dish_d_m = make_spin(QDoubleSpinBox(), 0.05, 50.0, 0.30, decimals=3, step=0.05, suffix=" m")
        self.sb_eff = make_spin(QDoubleSpinBox(), 0.05, 0.95, 0.60, decimals=2, step=0.05, suffix="")
        self.sb_mispoint_deg = make_spin(QDoubleSpinBox(), 0.0, 60.0, 0.0, decimals=2, step=0.1, suffix=" deg")

        # --- READ-ONLY BOXES ---
        self.sb_hpbw_ro = make_ro_box(decimals=4, suffix=" deg")
        self.sb_point_loss_ro = make_ro_box(decimals=4, suffix=" dB")
        self.sb_dish_gain_ro = make_ro_box(decimals=4, suffix=" dBi")

        self.cb_use_dish_as_gtx = QCheckBox("Use calculated dish gain as Tx gain (override Gtx)")
        self.cb_use_dish_as_gtx.setChecked(False)
        self.cb_use_dish_as_gtx.setToolTip(
            "When enabled, Gtx follows the calculated dish gain (useful for satellite link budgets / theory validation)."
        )

        g2.addWidget(QLabel("UE Gain Gtx:"), 0, 0)
        g2.addWidget(self.sb_ue_gain, 0, 1)
        g2.addWidget(QLabel("gNB Gain Grx:"), 1, 0)
        g2.addWidget(self.sb_gnb_gain, 1, 1)
        g2.addWidget(QLabel("Tx Loss Ltx:"), 2, 0)
        g2.addWidget(self.sb_cable_loss, 2, 1)

        g2.addWidget(QLabel("Dish D:"), 3, 0)
        g2.addWidget(self.sb_dish_d_m, 3, 1)
        g2.addWidget(QLabel("Efficiency η:"), 4, 0)
        g2.addWidget(self.sb_eff, 4, 1)
        g2.addWidget(QLabel("Mispoint θ:"), 5, 0)
        g2.addWidget(self.sb_mispoint_deg, 5, 1)

        g2.addWidget(QLabel("HPBW (deg):"), 6, 0)
        g2.addWidget(self.sb_hpbw_ro, 6, 1)
        g2.addWidget(QLabel("Pointing Loss Lp:"), 7, 0)
        g2.addWidget(self.sb_point_loss_ro, 7, 1)
        g2.addWidget(QLabel("Dish Gain (calc):"), 8, 0)
        g2.addWidget(self.sb_dish_gain_ro, 8, 1)

        g2.addWidget(self.cb_use_dish_as_gtx, 9, 0, 1, 2)

        box_flow = QGroupBox("Uplink Flow (Visual)")
        flow = QHBoxLayout(box_flow)
        flow.setSpacing(10)
        self.flow_boxes = []
        for t in ["Source", "Mod", "RF", "Antenna", "Channel", "gNB"]:
            card = QFrame()
            card.setObjectName("FlowCard")
            card.setFrameShape(QFrame.StyledPanel)
            v = QVBoxLayout(card)
            v.setContentsMargins(12, 12, 12, 12)
            v.setSpacing(8)
            title = QLabel(t)
            title.setObjectName("FlowTitle")
            title.setAlignment(Qt.AlignCenter)
            v.addWidget(title)
            st = QLabel("OK")
            st.setObjectName("FlowStatus")
            st.setAlignment(Qt.AlignCenter)
            v.addWidget(st)
            self.flow_boxes.append((card, st))
            flow.addWidget(card)

        layout.addWidget(box_tx)
        layout.addWidget(box_ant)
        layout.addWidget(box_flow)
        layout.addStretch()
        self.stack.addWidget(page)

    def build_environment_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        box = QGroupBox("Environment / Weather (Impairments)")
        g = QGridLayout(box)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

        self.cb_rain = QCheckBox("Rain")
        self.sb_rain = make_spin(QDoubleSpinBox(), 0.0, 300.0, 10.0, decimals=2, step=1.0, suffix=" mm/h")

        self.cb_snow = QCheckBox("Snow")
        self.sb_snow = make_spin(QDoubleSpinBox(), 0.0, 100.0, 2.0, decimals=2, step=0.5, suffix=" mm/h")

        self.cb_wind = QCheckBox("Wind")
        self.sb_wind = make_spin(QDoubleSpinBox(), 0.0, 200.0, 5.0, decimals=2, step=0.5, suffix=" m/s")

        self.cb_fog = QCheckBox("Fog")
        self.sb_fog = make_spin(QDoubleSpinBox(), 0.0, 10.0, 0.2, decimals=3, step=0.05, suffix=" g/m³")

        self.cb_humidity = QCheckBox("Humidity")
        self.sb_humidity = make_spin(QDoubleSpinBox(), 0.0, 100.0, 60.0, decimals=1, step=1.0, suffix=" %")

        self.cb_dust = QCheckBox("Dust / Sand")
        self.sb_dust = make_spin(QDoubleSpinBox(), 0.0, 50.0, 0.0, decimals=2, step=0.5, suffix=" level")

        self.cb_temp = QCheckBox("Extreme Temperature")
        self.sb_temp = make_spin(QDoubleSpinBox(), -80.0, 120.0, 20.0, decimals=1, step=1.0, suffix=" °C")

        g.addWidget(self.cb_rain, 0, 0)
        g.addWidget(self.sb_rain, 0, 1)
        g.addWidget(self.cb_snow, 1, 0)
        g.addWidget(self.sb_snow, 1, 1)
        g.addWidget(self.cb_wind, 2, 0)
        g.addWidget(self.sb_wind, 2, 1)
        g.addWidget(self.cb_fog, 3, 0)
        g.addWidget(self.sb_fog, 3, 1)
        g.addWidget(self.cb_humidity, 4, 0)
        g.addWidget(self.sb_humidity, 4, 1)
        g.addWidget(self.cb_dust, 5, 0)
        g.addWidget(self.sb_dust, 5, 1)
        g.addWidget(self.cb_temp, 6, 0)
        g.addWidget(self.sb_temp, 6, 1)

        hint = QLabel("Models are simplified (educational). Standards are listed in References.")
        hint.setObjectName("HintLabel")

        layout.addWidget(box)
        layout.addWidget(hint)
        layout.addStretch()
        self.stack.addWidget(page)

    def build_budget_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        box_in = QGroupBox("Link Budget Inputs")
        g = QGridLayout(box_in)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

        self.sb_freq = make_spin(QDoubleSpinBox(), 1.0, 300000.0, 3500.0, decimals=0, step=50.0, suffix=" MHz")
        self.sb_dist = make_spin(QDoubleSpinBox(), 0.001, 50000.0, 2.0, decimals=3, step=0.1, suffix=" km")
        self.sb_nf = make_spin(QDoubleSpinBox(), 0.0, 30.0, 5.0, decimals=2, step=0.5, suffix=" dB")
        self.sb_misc = make_spin(QDoubleSpinBox(), 0.0, 200.0, 2.0, decimals=2, step=0.5, suffix=" dB")

        g.addWidget(QLabel("Frequency f:"), 0, 0)
        g.addWidget(self.sb_freq, 0, 1)
        g.addWidget(QLabel("Distance d:"), 1, 0)
        g.addWidget(self.sb_dist, 1, 1)
        g.addWidget(QLabel("Noise Figure NF:"), 2, 0)
        g.addWidget(self.sb_nf, 2, 1)
        g.addWidget(QLabel("Extra losses Lmisc:"), 3, 0)
        g.addWidget(self.sb_misc, 3, 1)

        box_hist = QGroupBox("Snapshots / History")
        v = QVBoxLayout(box_hist)
        v.setSpacing(10)

        btn_row = QHBoxLayout()
        self.btn_snapshot = QPushButton("Add Snapshot")
        self.btn_export_csv = QPushButton("Export CSV")
        self.btn_reset = QPushButton("Reset to Defaults")
        btn_row.addWidget(self.btn_snapshot)
        btn_row.addWidget(self.btn_export_csv)
        btn_row.addWidget(self.btn_reset)
        btn_row.addStretch(1)
        v.addLayout(btn_row)

        self.tbl_hist = QTableWidget(0, 8)
        self.tbl_hist.setHorizontalHeaderLabels([
            "Time", "f (MHz)", "d (km)", "EIRP (dBm)", "Pr (dBm)", "SNR (dB)", "Eb/N0 (dB)", "Status"
        ])
        self.tbl_hist.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        v.addWidget(self.tbl_hist)

        layout.addWidget(box_in)
        layout.addWidget(box_hist)
        layout.addStretch()
        self.stack.addWidget(page)

    def build_visualization_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        self.viz_tabs = QTabWidget()
        layout.addWidget(self.viz_tabs)

        # ✅ optional but safe: helps prevent weird stretching
        self.viz_tabs.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Waveform ---
        wave_tab = QWidget()
        wave_layout = QVBoxLayout(wave_tab)
        self.wave_plot = pg.PlotWidget()
        self.wave_curve = self.wave_plot.plot([], [])
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Samples")
        wave_layout.addWidget(self.wave_plot)
        self.viz_tabs.addTab(wave_tab, "Waveform")

        # --- Constellation ---
        const_tab = QWidget()
        const_layout = QVBoxLayout(const_tab)
        self.const_plot = pg.PlotWidget()
        self.const_plot.setAspectLocked(True)
        self.const_plot.showGrid(x=True, y=True, alpha=0.2)
        self.const_plot.setLabel("left", "Q")
        self.const_plot.setLabel("bottom", "I")
        self.const_scatter = pg.ScatterPlotItem(size=5, pen=None)
        self.const_plot.addItem(self.const_scatter)
        const_layout.addWidget(self.const_plot)
        self.viz_tabs.addTab(const_tab, "Constellation")

        # --- Waterfall ---
        wf_tab = QWidget()
        wf_layout = QVBoxLayout(wf_tab)
        self.wf_plot = pg.PlotWidget()
        self.wf_plot.showGrid(x=True, y=True, alpha=0.2)
        self.wf_plot.setLabel("left", "dB (dBm scale)")
        wf_layout.addWidget(self.wf_plot)
        self.viz_tabs.addTab(wf_tab, "Waterfall")

        # --- Coverage Map (offline) ---
        cov_tab = QWidget()
        cov_layout = QVBoxLayout(cov_tab)
        cov_layout.setSpacing(10)

        row = QHBoxLayout()
        self.btn_load_map = QPushButton("Load Offline Earth Raster")
        self.btn_clear_map = QPushButton("Clear Background")
        row.addWidget(self.btn_load_map)
        row.addWidget(self.btn_clear_map)
        row.addStretch(1)
        cov_layout.addLayout(row)

        self.map_plot = pg.PlotWidget()
        self.map_plot.showGrid(x=True, y=True, alpha=0.2)
        self.map_plot.setLabel("left", "Latitude (deg)")
        self.map_plot.setLabel("bottom", "Longitude (deg)")
        self.map_plot.setXRange(-180, 180)
        self.map_plot.setYRange(-90, 90)

        lon = np.linspace(-180, 180, 200)
        for lat_line in [-60, -30, 0, 30, 60]:
            self.map_plot.plot(lon, np.full_like(lon, lat_line), pen=pg.mkPen((80, 80, 80), width=1))
        lat = np.linspace(-90, 90, 200)
        for lon_line in [-120, -60, 0, 60, 120]:
            self.map_plot.plot(np.full_like(lat, lon_line), lat, pen=pg.mkPen((80, 80, 80), width=1))

        self.coverage_item = pg.PlotDataItem([], [], pen=pg.mkPen("g", width=2))
        self.coverage_center = pg.ScatterPlotItem(size=10, pen=None, brush=pg.mkBrush("w"))
        self.map_plot.addItem(self.coverage_item)
        self.map_plot.addItem(self.coverage_center)

        # Satellite subpoint marker (optional, only when TLE enabled)
        self.sat_subpoint = pg.ScatterPlotItem(
            size=9, pen=pg.mkPen((255, 255, 255), width=1),
            brush=pg.mkBrush(160, 210, 255, 200)
        )
        self.map_plot.addItem(self.sat_subpoint)

        cov_layout.addWidget(self.map_plot)
        self.viz_tabs.addTab(cov_tab, "Coverage Map")

        # --- Link View (2.5D) ---
        link_tab = QWidget()
        link_layout = QVBoxLayout(link_tab)

        self.link_plot = pg.PlotWidget()
        self.link_plot.setAspectLocked(True)
        self.link_plot.showGrid(x=True, y=True, alpha=0.2)
        self.link_plot.setXRange(-10, 10)
        self.link_plot.setYRange(-3, 22)

        # Base + satellite + shadow
        self.bs_item = pg.ScatterPlotItem(size=16, pen=None, brush=pg.mkBrush("w"))
        self.sat_item = pg.ScatterPlotItem(size=16, pen=None, brush=pg.mkBrush((180, 200, 255)))
        self.sat_shadow = pg.ScatterPlotItem(size=18, pen=None, brush=pg.mkBrush(0, 0, 0, 80))
        self.link_plot.addItem(self.bs_item)
        self.link_plot.addItem(self.sat_shadow)
        self.link_plot.addItem(self.sat_item)

        # Beam line + cone outline
        self.beam_line = pg.PlotDataItem([], [], pen=pg.mkPen("g", width=3))
        self.beam_cone = pg.PlotDataItem([], [], pen=pg.mkPen("g", width=2))
        self.link_plot.addItem(self.beam_line)
        self.link_plot.addItem(self.beam_cone)

        # Text
        self.link_text = pg.TextItem("", anchor=(0, 0))
        self.link_plot.addItem(self.link_text)

        # Earth horizon arc
        self.earth_arc = pg.PlotDataItem([], [], pen=pg.mkPen((120, 120, 120), width=2))
        self.link_plot.addItem(self.earth_arc)

        # Orbit arc (dynamic)
        self.orbit_arc = pg.PlotDataItem([], [], pen=pg.mkPen((80, 160, 255), width=2))
        self.link_plot.addItem(self.orbit_arc)

        # Beam fill (semi-transparent)
        self.beam_fill = pg.PlotDataItem([], [], pen=None)
        self.link_plot.addItem(self.beam_fill)

        # Animated packets
        self.packet_item = pg.ScatterPlotItem(size=6, pen=None, brush=pg.mkBrush(255, 255, 255, 200))
        self.link_plot.addItem(self.packet_item)

        # Satellite glow
        self.sat_glow = pg.ScatterPlotItem(size=40, pen=None, brush=pg.mkBrush(120, 170, 255, 70))
        self.link_plot.addItem(self.sat_glow)

        # Trail + ground track
        self.trail_item = pg.PlotDataItem([], [], pen=pg.mkPen((180, 200, 255), width=2))
        self.link_plot.addItem(self.trail_item)
        self.groundtrack_item = pg.PlotDataItem([], [], pen=pg.mkPen((140, 140, 140), width=1))
        self.link_plot.addItem(self.groundtrack_item)

        link_layout.addWidget(self.link_plot)
        self.viz_tabs.addTab(link_tab, "Link View (2.5D)")

        # --- NEW: Live Metrics (scrollable) ---
        metrics_tab = QWidget()
        metrics_outer = QVBoxLayout(metrics_tab)
        metrics_outer.setContentsMargins(0, 0, 0, 0)

        metrics_scroll = QScrollArea()
        metrics_scroll.setWidgetResizable(True)
        metrics_scroll.setFrameShape(QFrame.NoFrame)
        metrics_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        metrics_content = QWidget()
        metrics_layout = QVBoxLayout(metrics_content)
        metrics_layout.setSpacing(10)
        metrics_layout.setContentsMargins(12, 12, 12, 12)

        # Key: stops compression, enables natural height and scroll
        metrics_layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        info = QLabel("Live metrics (rolling window). Updates automatically as you change inputs.")
        info.setObjectName("HintLabel")
        metrics_layout.addWidget(info)

        self.live_plot_pr = pg.PlotWidget()
        self.live_plot_pr.setLabel("left", "Pr (dBm)")
        self.live_plot_pr.setLabel("bottom", "Time (s)")
        self.live_plot_pr.showGrid(x=True, y=True, alpha=0.2)
        self.live_plot_pr.setMinimumHeight(260)  # ✅ important (prevents squish)
        self.live_curve_pr = self.live_plot_pr.plot([], [])

        self.live_plot_snr = pg.PlotWidget()
        self.live_plot_snr.setLabel("left", "SNR (dB)")
        self.live_plot_snr.setLabel("bottom", "Time (s)")
        self.live_plot_snr.showGrid(x=True, y=True, alpha=0.2)
        self.live_plot_snr.setMinimumHeight(260)
        self.live_curve_snr = self.live_plot_snr.plot([], [])

        self.live_plot_ebn0 = pg.PlotWidget()
        self.live_plot_ebn0.setLabel("left", "Eb/N0 & Margin (dB)")
        self.live_plot_ebn0.setLabel("bottom", "Time (s)")
        self.live_plot_ebn0.showGrid(x=True, y=True, alpha=0.2)
        self.live_plot_ebn0.setMinimumHeight(260)
        self.live_curve_ebn0 = self.live_plot_ebn0.plot([], [])
        self.live_curve_margin = self.live_plot_ebn0.plot([], [])

        metrics_layout.addWidget(self.live_plot_pr)
        metrics_layout.addWidget(self.live_plot_snr)
        metrics_layout.addWidget(self.live_plot_ebn0)

        metrics_layout.addStretch(1)

        metrics_scroll.setWidget(metrics_content)
        metrics_outer.addWidget(metrics_scroll)

        self.viz_tabs.addTab(metrics_tab, "Live Metrics")

        layout.addStretch()
        self.stack.addWidget(page)

    def build_scenarios_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)

        self.cb_scenario = QComboBox()
        self.cb_scenario.addItems([
            "Urban LOS", "Urban NLOS", "Rural LOS", "Indoor",
            "Satellite-like (LEO demo)", "Theory 2.5.6 (Uplink clear-sky)", "Theory 2.5.7 (Uplink + rain)"
        ])

        btn = QPushButton("Apply Scenario")
        btn.clicked.connect(self.apply_scenario)

        layout.addWidget(QLabel("Scenario Presets"))
        layout.addWidget(self.cb_scenario)
        layout.addWidget(btn)
        layout.addStretch()
        self.stack.addWidget(page)

    # ✅ REPLACED build_advanced_page() (scroll + policies + correct stretch)
    def build_advanced_page(self):
        # Outer page
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scroll area (so content never gets compressed)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Inner content widget (THIS is what scrolls)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Key: prevents weird compression when toggling sections
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)

        scroll.setWidget(content)
        outer.addWidget(scroll)

        # ================== YOUR EXISTING CONTENT (unchanged) ==================

        self.cb_enable_adv = QCheckBox("Enable Advanced Settings (OFF by default)")
        self.cb_enable_adv.setChecked(False)

        adv_box = QGroupBox("Advanced Settings (Optional)")
        g = QGridLayout(adv_box)
        g.setHorizontalSpacing(12)
        g.setVerticalSpacing(10)

        self.sb_lat = make_spin(QDoubleSpinBox(), -90.0, 90.0, 35.0, decimals=3, step=0.1, suffix=" deg")
        self.sb_lon = make_spin(QDoubleSpinBox(), -180.0, 180.0, 25.0, decimals=3, step=0.1, suffix=" deg")
        self.sb_sat_alt_km = make_spin(QDoubleSpinBox(), 200.0, 36000.0, 550.0, decimals=2, step=50.0, suffix=" km")
        self.sb_elev_deg = make_spin(QDoubleSpinBox(), 5.0, 90.0, 30.0, decimals=2, step=1.0, suffix=" deg")
        self.sb_pol_loss = make_spin(QDoubleSpinBox(), 0.0, 50.0, 0.0, decimals=2, step=0.5, suffix=" dB")
        self.sb_atm_loss = make_spin(QDoubleSpinBox(), 0.0, 100.0, 0.0, decimals=2, step=0.5, suffix=" dB")

        g.addWidget(QLabel("Latitude:"), 0, 0)
        g.addWidget(self.sb_lat, 0, 1)
        g.addWidget(QLabel("Longitude:"), 1, 0)
        g.addWidget(self.sb_lon, 1, 1)
        g.addWidget(QLabel("Satellite altitude (fallback):"), 2, 0)
        g.addWidget(self.sb_sat_alt_km, 2, 1)
        g.addWidget(QLabel("Elevation angle (fallback):"), 3, 0)
        g.addWidget(self.sb_elev_deg, 3, 1)
        g.addWidget(QLabel("Polarization loss:"), 4, 0)
        g.addWidget(self.sb_pol_loss, 4, 1)
        g.addWidget(QLabel("Extra atmospheric loss:"), 5, 0)
        g.addWidget(self.sb_atm_loss, 5, 1)

        # --- Real satellite (TLE) optional controls ---
        self.cb_use_tle = QCheckBox("Use real satellite (TLE / live orbit) – optional")
        self.cb_use_tle.setChecked(False)
        self.cb_use_tle.setEnabled(SKYFIELD_OK)

        self.cb_celestrak_group = QComboBox()
        self.cb_celestrak_group.addItems([
            "ISS (ZARYA) (from stations)",
            "Stations (active)",
            "Starlink",
            "GPS (operational)",
            "Weather"
        ])
        self.cb_celestrak_group.setEnabled(SKYFIELD_OK)

        self.btn_load_tle = QPushButton("Load TLE (File)")
        self.btn_fetch_tle = QPushButton("Fetch TLE (CelesTrak)")
        self.btn_fetch_tle.setEnabled(SKYFIELD_OK)

        self.cb_sat_select = QComboBox()
        self.cb_sat_select.setEnabled(False)

        self.lbl_tle_status = QLabel("TLE: not loaded")
        self.lbl_tle_status.setObjectName("HintLabel")

        g.addWidget(self.cb_use_tle, 6, 0, 1, 2)
        g.addWidget(QLabel("CelesTrak Group:"), 7, 0)
        g.addWidget(self.cb_celestrak_group, 7, 1)

        row_btns = QHBoxLayout()
        row_btns.addWidget(self.btn_load_tle)
        row_btns.addWidget(self.btn_fetch_tle)
        row_btns.addStretch(1)
        g.addLayout(row_btns, 8, 0, 1, 2)

        g.addWidget(QLabel("Satellite:"), 9, 0)
        g.addWidget(self.cb_sat_select, 9, 1)
        g.addWidget(self.lbl_tle_status, 10, 0, 1, 2)

        # --- Satellite Theory (G/T) validation ---
        self.cb_theory_mode = QCheckBox("Enable Satellite Theory Mode (G/T, k=228.6)")
        self.cb_theory_mode.setChecked(False)
        self.cb_theory_mode.setToolTip("Uses satellite-style link budget: C/N0 = EIRP(dBW) - L_u + (G/T) + 228.6")

        theory_box = QGroupBox("Satellite Theory Parameters (Receiver + Atmosphere)")
        tg = QGridLayout(theory_box)
        tg.setHorizontalSpacing(12)
        tg.setVerticalSpacing(10)

        self.sb_theory_theta3db = make_spin(QDoubleSpinBox(), 0.1, 60.0, 2.0, decimals=3, step=0.1, suffix=" deg")
        self.sb_theory_eta_rx = make_spin(QDoubleSpinBox(), 0.05, 0.95, 0.55, decimals=2, step=0.05, suffix="")
        self.sb_theory_lr = make_spin(QDoubleSpinBox(), 0.0, 30.0, 3.0, decimals=2, step=0.5, suffix=" dB")
        self.sb_theory_lfrx = make_spin(QDoubleSpinBox(), 0.0, 20.0, 1.0, decimals=2, step=0.1, suffix=" dB")
        self.sb_theory_frx = make_spin(QDoubleSpinBox(), 0.0, 20.0, 3.0, decimals=2, step=0.1, suffix=" dB")
        self.sb_theory_ta = make_spin(QDoubleSpinBox(), 1.0, 5000.0, 290.0, decimals=1, step=10.0, suffix=" K")
        self.sb_theory_tf = make_spin(QDoubleSpinBox(), 1.0, 5000.0, 290.0, decimals=1, step=10.0, suffix=" K")

        self.sb_theory_atm = make_spin(QDoubleSpinBox(), 0.0, 100.0, 0.3, decimals=2, step=0.1, suffix=" dB")
        self.sb_theory_rain_db = make_spin(QDoubleSpinBox(), 0.0, 100.0, 0.0, decimals=2, step=0.5, suffix=" dB")
        self.sb_theory_pol = make_spin(QDoubleSpinBox(), 0.0, 20.0, 0.0, decimals=2, step=0.5, suffix=" dB")

        self.lbl_theory_gt = QLabel("-")
        self.lbl_theory_cn0 = QLabel("-")
        self.lbl_theory_gt.setObjectName("ResultValue")
        self.lbl_theory_cn0.setObjectName("ResultValue")

        self.txt_theory_validation = QPlainTextEdit()
        self.txt_theory_validation.setReadOnly(True)
        self.txt_theory_validation.setMaximumHeight(120)
        self.txt_theory_validation.setPlaceholderText("Validation output will appear here when a theory scenario is selected.")

        r = 0
        tg.addWidget(QLabel("Rx θ3dB:"), r, 0); tg.addWidget(self.sb_theory_theta3db, r, 1)
        tg.addWidget(QLabel("Rx efficiency η:"), r, 2); tg.addWidget(self.sb_theory_eta_rx, r, 3)
        r += 1
        tg.addWidget(QLabel("Coverage edge loss LR:"), r, 0); tg.addWidget(self.sb_theory_lr, r, 1)
        tg.addWidget(QLabel("Feeder loss LFRX:"), r, 2); tg.addWidget(self.sb_theory_lfrx, r, 3)
        r += 1
        tg.addWidget(QLabel("Receiver NF FRX:"), r, 0); tg.addWidget(self.sb_theory_frx, r, 1)
        tg.addWidget(QLabel("Antenna noise TA:"), r, 2); tg.addWidget(self.sb_theory_ta, r, 3)
        r += 1
        tg.addWidget(QLabel("Feeder temp TF:"), r, 0); tg.addWidget(self.sb_theory_tf, r, 1)
        tg.addWidget(QLabel("Pol loss LPOL:"), r, 2); tg.addWidget(self.sb_theory_pol, r, 3)
        r += 1
        tg.addWidget(QLabel("Atm loss LA (clear):"), r, 0); tg.addWidget(self.sb_theory_atm, r, 1)
        tg.addWidget(QLabel("Rain attenuation (dB):"), r, 2); tg.addWidget(self.sb_theory_rain_db, r, 3)
        r += 1
        tg.addWidget(QLabel("Computed (G/T):"), r, 0); tg.addWidget(self.lbl_theory_gt, r, 1)
        tg.addWidget(QLabel("Computed (C/N0):"), r, 2); tg.addWidget(self.lbl_theory_cn0, r, 3)
        r += 1
        tg.addWidget(QLabel("Validation (Examples 2.5.6 / 2.5.7):"), r, 0, 1, 4)
        r += 1
        tg.addWidget(self.txt_theory_validation, r, 0, 1, 4)

        theory_hint = QLabel(
            "Theory mode uses: C/N0(dBHz) = EIRP(dBW) - L_u + (G/T) + 228.6\n"
            "Where: L_u = FSPL + LA(clear+rain) + LPOL, and (G/T) is derived from θ3dB, η, LR, LFRX, FRX, TA, TF."
        )
        theory_hint.setObjectName("HintLabel")

        note = QLabel(
            "Advanced is OFF by default.\n"
            "When enabled: adds extra losses + may replace distance with a slant-range.\n"
            "Optional: Real satellites (TLE via Skyfield/SGP4) + offline cache for fetched TLE.\n"
        )
        note.setObjectName("HintLabel")

        # ✅ Critical: prevent the groupboxes from getting vertically squished
        adv_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        theory_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        layout.addWidget(self.cb_enable_adv)
        layout.addWidget(adv_box)

        layout.addWidget(self.cb_theory_mode)
        layout.addWidget(theory_box)
        layout.addWidget(theory_hint)
        layout.addWidget(note)

        # ✅ Only ONE stretch at the very end
        layout.addStretch(1)

        self.adv_box = adv_box
        self.adv_box.setEnabled(False)

        self.theory_box = theory_box
        self.theory_box.setEnabled(False)

        self.stack.addWidget(page)

    def build_formulas_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(
            "CORE FORMULAS (with units)\n\n"
            "1) EIRP\n"
            "EIRP(dBm) = Ptx(dBm) + Gtx(dBi) - Ltx(dB)\n"
            "EIRP(dBW) = EIRP(dBm) - 30\n\n"
            "2) Free Space Path Loss (FSPL)\n"
            "FSPL(dB) = 32.45 + 20log10(f_MHz) + 20log10(d_km)\n\n"
            "3) Received Power (simple link budget)\n"
            "Pr(dBm) = EIRP(dBm) + Grx(dBi) - [FSPL + Lweather + Lmisc + Lpoint + ...]\n\n"
            "4) Noise / Thermal\n"
            "Noise(dBm) = -174(dBm/Hz) + 10log10(B_Hz) + NF(dB)\n"
            "N0(dBm/Hz) ≈ -174 + NF\n\n"
            "5) SNR\n"
            "SNR(dB) = Pr(dBm) - Noise(dBm)\n\n"
            "6) C/N0 and Eb/N0\n"
            "C/N0(dB-Hz) = Pr(dBm) - N0(dBm/Hz)\n"
            "Eb/N0(dB) = C/N0(dB-Hz) - 10log10(Rb_Hz)\n\n"
            "7) Dish Gain & Beamwidth (parabolic approx)\n"
            "λ = c / f\n"
            "G(dBi) ≈ 10log10( η * (πD/λ)^2 )\n"
            "HPBW(deg) ≈ 70 * (λ / D)\n\n"
            "8) Pointing Loss (rule-of-thumb)\n"
            "Lp(dB) ≈ 12 * (θ / HPBW)^2\n\n"
            "9) Power conversions\n"
            "P(W) = 10^((P(dBm)-30)/10)\n"
            "P(dBm) = 10log10(P(W)*1000)\n\n"
            "Notes:\n"
            "- dB-Hz: noise-normalized carrier ratio per Hz.\n"
        )

        layout.addWidget(txt)
        layout.addStretch()
        self.stack.addWidget(page)

    def build_references_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        txt = QLabel(
            "References (static, offline)\n\n"
            "• ITU-R P.618: Earth-space propagation impairments (rain/cloud/scintillation)\n"
            "• ITU-R P.676: Atmospheric gases attenuation\n"
            "• ITU-R S.465: Reference earth station antenna patterns\n"
            "• ITU-R S.1528: Satellite antenna radiation patterns\n\n"
            "Extra (optional tools/data):\n"
            "• Skyfield + SGP4: Satellite propagation from TLE (real orbits)\n"
            "• CelesTrak: Public TLE sets (ISS/Starlink/GPS/Weather/etc.)\n"
            "• Natural Earth: Public-domain offline basemaps (raster/vector)\n\n"
            "This simulator uses simplified educational models.\n"
        )
        txt.setStyleSheet("font-size: 12px;")
        txt.setWordWrap(True)

        layout.addWidget(txt)
        layout.addStretch()
        self.stack.addWidget(page)

    # ================== GLOBAL RIGHT DOCK ==================

    def build_global_results_dock(self):
        dock = QDockWidget("Global Results (Live)", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(12, 12, 12, 12)
        v.setSpacing(12)

        # Pretty status "card"
        self.g_status = QLabel("—")
        self.g_status.setAlignment(Qt.AlignCenter)
        self.g_status.setObjectName("StatusBadge")
        v.addWidget(self.g_status)

        grid = QGridLayout()
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)

        self.g_pr = QLabel("-")
        self.g_snr = QLabel("-")
        self.g_mcs = QLabel("-")
        self.g_eirp = QLabel("-")
        self.g_cn0 = QLabel("-")
        self.g_ebn0 = QLabel("-")
        self.g_fspl = QLabel("-")
        self.g_weather = QLabel("-")
        self.g_losses = QLabel("-")
        self.g_margin = QLabel("-")

        items = [
            ("Pr (dBm)", self.g_pr),
            ("SNR (dB)", self.g_snr),
            ("MCS", self.g_mcs),
            ("EIRP (dBm)", self.g_eirp),
            ("C/N0 (dB-Hz)", self.g_cn0),
            ("Eb/N0 (dB)", self.g_ebn0),
            ("FSPL (dB)", self.g_fspl),
            ("Weather Loss (dB)", self.g_weather),
            ("Total Losses (dB)", self.g_losses),
            ("Margin (dB)", self.g_margin),
        ]
        for i, (lab, val) in enumerate(items):
            labw = QLabel(lab + ":")
            labw.setObjectName("ResultLabel")
            val.setObjectName("ResultValue")
            grid.addWidget(labw, i, 0)
            grid.addWidget(val, i, 1)
        v.addLayout(grid)

        v.addWidget(QLabel("Eb/N0 Bar"))
        self.bar_ebn0 = QProgressBar()
        self.bar_ebn0.setRange(0, 100)
        v.addWidget(self.bar_ebn0)

        v.addWidget(QLabel("Link Quality"))
        self.bar_quality = QProgressBar()
        self.bar_quality.setRange(0, 100)
        v.addWidget(self.bar_quality)

        self.g_explain = QLabel("")
        self.g_explain.setWordWrap(True)
        self.g_explain.setObjectName("HintLabel")
        v.addWidget(QLabel("Auto-Explain"))
        v.addWidget(self.g_explain)

        dock.setWidget(w)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ================== SIGNALS ==================

    def connect_all_signals(self):
        # boxed nav
        self.nav_group.buttonClicked[int].connect(self.stack.setCurrentIndex)

        self.btn_snapshot.clicked.connect(self.add_snapshot)
        self.btn_export_csv.clicked.connect(self.export_csv)
        self.btn_reset.clicked.connect(self.reset_defaults)

        self.cb_mod.currentIndexChanged.connect(self.update_results)
        self.cb_scenario.currentIndexChanged.connect(self.update_results)

        for sb in [
            self.sb_bw_mhz, self.sb_tx_pwr_dbm, self.sb_rb_mbps,
            self.sb_ue_gain, self.sb_gnb_gain, self.sb_cable_loss,
            self.sb_dish_d_m, self.sb_eff, self.sb_mispoint_deg,
            self.sb_rain, self.sb_snow, self.sb_wind, self.sb_fog,
            self.sb_humidity, self.sb_dust, self.sb_temp,
            self.sb_freq, self.sb_dist, self.sb_nf, self.sb_misc,
            self.sb_lat, self.sb_lon, self.sb_sat_alt_km, self.sb_elev_deg,
            self.sb_pol_loss, self.sb_atm_loss,
            self.sb_theory_theta3db, self.sb_theory_eta_rx, self.sb_theory_lr,
            self.sb_theory_lfrx, self.sb_theory_frx, self.sb_theory_ta, self.sb_theory_tf,
            self.sb_theory_atm, self.sb_theory_rain_db, self.sb_theory_pol
        ]:
            sb.valueChanged.connect(self.update_results)

        for cb in [
            self.cb_rain, self.cb_snow, self.cb_wind, self.cb_fog,
            self.cb_humidity, self.cb_dust, self.cb_temp,
            self.cb_enable_adv
        ]:
            cb.stateChanged.connect(self.on_checkbox_changed)

        self.cb_use_dish_as_gtx.stateChanged.connect(self.on_use_dish_as_gtx_changed)
        self.cb_theory_mode.stateChanged.connect(self.on_theory_mode_changed)

        # TLE controls
        self.cb_use_tle.stateChanged.connect(self.on_checkbox_changed)
        self.btn_load_tle.clicked.connect(self.load_tle_file)
        self.btn_fetch_tle.clicked.connect(self.fetch_tle_from_celestrak)
        self.cb_sat_select.currentIndexChanged.connect(self.on_satellite_selection_changed)

        # Offline map background
        self.btn_load_map.clicked.connect(self.load_offline_map_raster)
        self.btn_clear_map.clicked.connect(self.clear_map_background)

    def on_checkbox_changed(self):
        self.adv_box.setEnabled(self.cb_enable_adv.isChecked())

        # TLE enable flag (safe)
        self.sat_tracker.set_enabled(self.cb_use_tle.isChecked())
        if not self.cb_use_tle.isChecked():
            self._sat_trail.clear()
            self.trail_item.setData([], [])
            self.groundtrack_item.setData([], [])
            self.sat_subpoint.setData([])
        self.update_results()

    def on_use_dish_as_gtx_changed(self):
        en = self.cb_use_dish_as_gtx.isChecked()
        self.sb_ue_gain.setEnabled(not en)
        if en:
            # keep UI consistent (copy calculated dish gain into Gtx field)
            try:
                self.sb_ue_gain.blockSignals(True)
                self.sb_ue_gain.setValue(float(self.sb_dish_gain_ro.value()))
            finally:
                self.sb_ue_gain.blockSignals(False)
        self.update_results()

    def on_theory_mode_changed(self):
        try:
            self.theory_box.setEnabled(self.cb_theory_mode.isChecked())
        except Exception:
            pass
        self.update_results()

    def on_satellite_selection_changed(self, idx):
        if not SKYFIELD_OK:
            return
        try:
            self.sat_tracker.set_active_index(int(idx))
            self._sat_trail.clear()
            self.update_results()
        except Exception:
            pass

    # ================== OPTIONAL LOADERS ==================

    def load_tle_file(self):
        if not SKYFIELD_OK:
            QMessageBox.information(self, "TLE", "Skyfield not installed. Install: pip install skyfield sgp4")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Open TLE file", "", "Text Files (*.txt);;All Files (*)")
        if not path:
            return

        try:
            txt = safe_read_text(path)
            tle_list = parse_tle_text(txt)
            ok, msg = self.sat_tracker.load_tle_list(tle_list)
            if ok:
                self.lbl_tle_status.setText(f"TLE loaded from file: {msg}")
                self.refresh_satellite_combo(limit=400)
            else:
                self.lbl_tle_status.setText(f"TLE error: {msg}")
            self.update_results()
        except Exception as e:
            QMessageBox.warning(self, "TLE", f"Failed to load TLE:\n{e}")

    def celestrak_url_for_group(self, group_label: str):
        base = "https://celestrak.org/NORAD/elements/gp.php"
        mapping = {
            "Stations (active)": "stations",
            "Starlink": "starlink",
            "GPS (operational)": "gps-ops",
            "Weather": "weather",
            "ISS (ZARYA) (from stations)": "stations",
        }
        group = mapping.get(group_label, "stations")
        return f"{base}?GROUP={group}&FORMAT=tle"

    def fetch_tle_from_celestrak(self):
        if not SKYFIELD_OK:
            QMessageBox.information(self, "CelesTrak", "Skyfield not installed. Install: pip install skyfield sgp4")
            return

        group_label = self.cb_celestrak_group.currentText()
        url = self.celestrak_url_for_group(group_label)

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "UplinkAntennaSimulator/1.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
            txt = data.decode("utf-8", errors="ignore")

            safe_name = group_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            cache_path = os.path.join(self.tle_cache_dir, f"{safe_name}.tle")
            with open(cache_path, "w", encoding="utf-8") as f:
                f.write(txt)

            tle_list = parse_tle_text(txt)

            if group_label.startswith("ISS"):
                iss = None
                for name, l1, l2 in tle_list:
                    if "ISS" in name.upper() and "ZARYA" in name.upper():
                        iss = [(name, l1, l2)]
                        break
                if iss is None:
                    for name, l1, l2 in tle_list:
                        if "ISS" in name.upper():
                            iss = [(name, l1, l2)]
                            break
                if iss is not None:
                    tle_list = iss

            ok, msg = self.sat_tracker.load_tle_list(tle_list)
            if ok:
                self.lbl_tle_status.setText(f"CelesTrak fetched ({group_label}) + cached: {msg}")
                limit = 300 if "Starlink" in group_label else 600
                self.refresh_satellite_combo(limit=limit)
            else:
                self.lbl_tle_status.setText(f"TLE error: {msg}")

            self.update_results()

        except Exception as e:
            group_label = self.cb_celestrak_group.currentText()
            safe_name = group_label.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
            cache_path = os.path.join(self.tle_cache_dir, f"{safe_name}.tle")
            if os.path.exists(cache_path):
                try:
                    txt = safe_read_text(cache_path)
                    tle_list = parse_tle_text(txt)

                    if group_label.startswith("ISS"):
                        iss = None
                        for name, l1, l2 in tle_list:
                            if "ISS" in name.upper() and "ZARYA" in name.upper():
                                iss = [(name, l1, l2)]
                                break
                        if iss is None:
                            for name, l1, l2 in tle_list:
                                if "ISS" in name.upper():
                                    iss = [(name, l1, l2)]
                                    break
                        if iss is not None:
                            tle_list = iss

                    ok, msg2 = self.sat_tracker.load_tle_list(tle_list)
                    if ok:
                        self.lbl_tle_status.setText(f"Online failed; loaded cached ({group_label}): {msg2}")
                        limit = 300 if "Starlink" in group_label else 600
                        self.refresh_satellite_combo(limit=limit)
                        self.update_results()
                        return
                except Exception:
                    pass

            QMessageBox.warning(self, "CelesTrak", f"Failed to fetch TLE (and cache not usable):\n{e}")

    def refresh_satellite_combo(self, limit=500):
        if not SKYFIELD_OK:
            return

        names = self.sat_tracker.names()
        if not names:
            self.cb_sat_select.clear()
            self.cb_sat_select.setEnabled(False)
            return

        self.cb_sat_select.blockSignals(True)
        self.cb_sat_select.clear()

        show = names[:max(1, int(limit))]
        for nm in show:
            self.cb_sat_select.addItem(nm)

        self.cb_sat_select.setEnabled(True)
        self.cb_sat_select.setCurrentIndex(0)
        self.cb_sat_select.blockSignals(False)

    def load_offline_map_raster(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Earth Raster Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All Files (*)"
        )
        if not path:
            return

        try:
            img = pg.QtGui.QImage(path)
            if img.isNull():
                raise ValueError("Could not load image.")

            img = img.convertToFormat(pg.QtGui.QImage.Format.Format_RGBA8888)
            w = img.width()
            h = img.height()

            ptr = img.bits()
            ptr.setsize(img.byteCount())
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))

            if self._map_bg_item is None:
                self._map_bg_item = pg.ImageItem()
                self.map_plot.addItem(self._map_bg_item)
                self._map_bg_item.setZValue(-10)

            self._map_bg_item.setImage(arr, autoLevels=False)

            tr = pg.QtGui.QTransform()
            tr.translate(-180.0, 90.0)
            tr.scale(360.0 / w, -180.0 / h)
            self._map_bg_item.setTransform(tr)

        except Exception as e:
            QMessageBox.warning(self, "Map Background", f"Failed to load raster:\n{e}")

    def clear_map_background(self):
        if self._map_bg_item is not None:
            self.map_plot.removeItem(self._map_bg_item)
            self._map_bg_item = None

    # ================== CORE LOGIC ==================

    def weather_loss_db(self, d_km, f_mhz):
        loss = 0.0
        if self.cb_rain.isChecked():
            loss += 0.02 * self.sb_rain.value() * d_km
        if self.cb_snow.isChecked():
            loss += 0.03 * self.sb_snow.value() * d_km
        if self.cb_fog.isChecked():
            loss += 0.5 * self.sb_fog.value() * d_km
        if self.cb_wind.isChecked():
            loss += max(0.0, (self.sb_wind.value() - 8.0) * 0.15)

        if self.cb_humidity.isChecked():
            hum = self.sb_humidity.value()
            freq_factor = clamp((f_mhz / 10000.0), 0.2, 3.0)
            loss += (hum / 100.0) * 0.6 * d_km * freq_factor

        if self.cb_dust.isChecked():
            loss += 0.8 * self.sb_dust.value() * d_km

        if self.cb_temp.isChecked():
            temp = self.sb_temp.value()
            if temp <= -10 or temp >= 40:
                loss += 0.5 * d_km

        return loss

    def suggest_mcs(self, snr):
        if snr < 0:
            return "QPSK"
        if snr < 8:
            return "QPSK / 16QAM"
        if snr < 15:
            return "16QAM"
        if snr < 22:
            return "64QAM"
        return "256QAM"

    def compute_slant_range_km(self):
        if SKYFIELD_OK and self.sat_tracker.enabled and self.sat_tracker.loaded:
            ok, lat, lon, alt_km, slant_km = self.sat_tracker.update(
                ground_lat_deg=float(self.sb_lat.value()),
                ground_lon_deg=float(self.sb_lon.value())
            )
            if ok and slant_km is not None:
                return max(1.0, float(slant_km))

        Re_km = 6371.0
        h_km = self.sb_sat_alt_km.value()
        elev = math.radians(self.sb_elev_deg.value())
        slant = math.sqrt((Re_km + h_km) ** 2 - (Re_km * math.cos(elev)) ** 2) - Re_km * math.sin(elev)
        return max(1.0, slant)

    def reset_defaults(self):
        self.cb_mod.setCurrentIndex(0)
        self.sb_bw_mhz.setValue(20.0)
        self.sb_tx_pwr_dbm.setValue(23.0)
        self.sb_rb_mbps.setValue(10.0)

        self.sb_ue_gain.setValue(2.0)
        self.sb_gnb_gain.setValue(18.0)
        self.sb_cable_loss.setValue(0.0)
        self.sb_dish_d_m.setValue(0.30)
        self.sb_eff.setValue(0.60)
        self.sb_mispoint_deg.setValue(0.0)

        self.cb_use_dish_as_gtx.setChecked(False)

        for cb in [self.cb_rain, self.cb_snow, self.cb_wind, self.cb_fog, self.cb_humidity, self.cb_dust, self.cb_temp]:
            cb.setChecked(False)

        self.sb_rain.setValue(10.0)
        self.sb_snow.setValue(2.0)
        self.sb_wind.setValue(5.0)
        self.sb_fog.setValue(0.2)
        self.sb_humidity.setValue(60.0)
        self.sb_dust.setValue(0.0)
        self.sb_temp.setValue(20.0)

        self.sb_freq.setValue(3500)
        self.sb_dist.setValue(2.0)
        self.sb_nf.setValue(5.0)
        self.sb_misc.setValue(2.0)

        self.cb_enable_adv.setChecked(False)
        self.sb_lat.setValue(35.0)
        self.sb_lon.setValue(25.0)
        self.sb_sat_alt_km.setValue(550.0)
        self.sb_elev_deg.setValue(30.0)
        self.sb_pol_loss.setValue(0.0)
        self.sb_atm_loss.setValue(0.0)

        # Satellite theory defaults (disabled)
        self.cb_theory_mode.setChecked(False)
        self.sb_theory_theta3db.setValue(2.0)
        self.sb_theory_eta_rx.setValue(0.55)
        self.sb_theory_lr.setValue(3.0)
        self.sb_theory_lfrx.setValue(1.0)
        self.sb_theory_frx.setValue(3.0)
        self.sb_theory_ta.setValue(290.0)
        self.sb_theory_tf.setValue(290.0)
        self.sb_theory_atm.setValue(0.3)
        self.sb_theory_rain_db.setValue(0.0)
        self.sb_theory_pol.setValue(0.0)
        try:
            self.txt_theory_validation.setPlainText('')
        except Exception:
            pass

        self.cb_use_tle.setChecked(False)
        self.lbl_tle_status.setText("TLE: not loaded")
        self.cb_sat_select.clear()
        self.cb_sat_select.setEnabled(False)

        self._sat_trail.clear()
        self._live_x.clear()
        self._live_pr.clear()
        self._live_snr.clear()
        self._live_ebn0.clear()
        self._live_margin.clear()
        self._live_t0 = None

        self.update_results()

    def auto_explain(self, snr_db, d_km, wloss_db, nf_db, lp_db):
        reasons = []
        if d_km > 10:
            reasons.append("μεγάλη απόσταση → FSPL↑")
        if wloss_db > 5:
            reasons.append("καιρός/κανάλι → απώλειες↑")
        if self.sb_tx_pwr_dbm.value() < 10:
            reasons.append("χαμηλή ισχύς εκπομπής")
        if lp_db > 1:
            reasons.append("mispoint → pointing loss↑")
        if nf_db > 8:
            reasons.append("NF↑ → θόρυβος↑")

        if snr_db < 0:
            quality = "κακή ποιότητα uplink"
        elif snr_db < 10:
            quality = "οριακή ποιότητα uplink"
        else:
            quality = "καλή ποιότητα uplink"

        if not reasons:
            reasons.append("ευνοϊκές συνθήκες")

        return f"Το σύστημα παρουσιάζει {quality} επειδή: " + ", ".join(reasons) + "."

    # ================== VISUAL UPDATES ==================
    # (everything below is unchanged from your file)

    def update_waveform(self, snr_db):
        n = 900
        t = np.arange(n)
        noise_std = 10 ** (-snr_db / 20.0)
        sig = np.sin(2 * np.pi * 0.02 * t)
        noisy = sig + np.random.normal(0, noise_std, size=n)
        self.wave_curve.setData(t, noisy)

    def constellation_points(self, mod_name, n_points, snr_db):
        rng = np.random.default_rng()
        if mod_name == "QPSK":
            const = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]], dtype=float) / math.sqrt(2)
        elif mod_name == "16QAM":
            xs = np.array([-3, -1, 1, 3], dtype=float)
            xv, yv = np.meshgrid(xs, xs)
            const = np.stack([xv.ravel(), yv.ravel()], axis=1)
            const = const / np.sqrt((const[:, 0] ** 2 + const[:, 1] ** 2).mean())
        elif mod_name == "64QAM":
            xs = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=float)
            xv, yv = np.meshgrid(xs, xs)
            const = np.stack([xv.ravel(), yv.ravel()], axis=1)
            const = const / np.sqrt((const[:, 0] ** 2 + const[:, 1] ** 2).mean())
        else:
            xs = np.arange(-15, 16, 2, dtype=float)
            xv, yv = np.meshgrid(xs, xs)
            const = np.stack([xv.ravel(), yv.ravel()], axis=1)
            const = const / np.sqrt((const[:, 0] ** 2 + const[:, 1] ** 2).mean())

        idx = rng.integers(0, len(const), size=n_points)
        pts = const[idx].copy()
        sigma = 10 ** (-(snr_db / 20.0))
        pts[:, 0] += rng.normal(0, sigma, size=n_points)
        pts[:, 1] += rng.normal(0, sigma, size=n_points)
        return pts

    def update_constellation(self, mod_name, snr_db):
        pts = self.constellation_points(mod_name, n_points=1200, snr_db=snr_db)
        spots = [{"pos": (pts[i, 0], pts[i, 1])} for i in range(len(pts))]
        self.const_scatter.setData(spots)
        self.const_plot.setXRange(-2, 2)
        self.const_plot.setYRange(-2, 2)

    def update_waterfall(self, eirp_dbm, grx_db, total_losses_db, pr_dbm):
        self.wf_plot.clear()
        labels = ["EIRP", "+Grx", "-Losses", "=Pr"]
        x = np.arange(len(labels))

        v1 = eirp_dbm
        v2 = grx_db
        v3 = -total_losses_db
        v4 = pr_dbm

        level1 = v1
        level2 = level1 + v2
        level3 = level2 + v3

        b1 = pg.BarGraphItem(x=[0], height=[v1], width=0.6, y0=[0])
        b2 = pg.BarGraphItem(x=[1], height=[v2], width=0.6, y0=[level1])
        b3 = pg.BarGraphItem(x=[2], height=[v3], width=0.6, y0=[level2])
        self.wf_plot.addItem(b1)
        self.wf_plot.addItem(b2)
        self.wf_plot.addItem(b3)

        self.wf_plot.plot([3], [v4], pen=None, symbol="o", symbolSize=10, symbolBrush="w")

        ax = self.wf_plot.getAxis("bottom")
        ax.setTicks([list(zip(x, labels))])
        self.wf_plot.setYRange(min(0, v4) - 50, max(level2, level1) + 20)

        for xi, yi, txt in [
            (0, level1, f"{v1:.1f}"),
            (1, level2, f"{v2:.1f}"),
            (2, level3, f"{v3:.1f}"),
            (3, v4, f"{v4:.1f}")
        ]:
            t = pg.TextItem(txt, anchor=(0.5, -0.3))
            t.setPos(xi, yi)
            self.wf_plot.addItem(t)

    def update_coverage_map(self, status_color):
        lat0 = float(self.sb_lat.value())
        lon0 = float(self.sb_lon.value())
        h_km = float(self.sb_sat_alt_km.value())
        elev = float(self.sb_elev_deg.value())
        base = clamp(h_km / 800.0, 0.3, 30.0)
        elev_factor = clamp((60.0 / max(elev, 5.0)), 1.0, 8.0)
        radius_deg = clamp(base * elev_factor, 2.0, 60.0)

        theta = np.linspace(0, 2*np.pi, 361)
        circle_lon = lon0 + radius_deg * np.cos(theta)
        circle_lat = lat0 + radius_deg * np.sin(theta)

        circle_lon = np.clip(circle_lon, -180, 180)
        circle_lat = np.clip(circle_lat, -90, 90)

        pen = pg.mkPen(status_color, width=2)
        self.coverage_item.setData(circle_lon, circle_lat, pen=pen)
        self.coverage_center.setData([{"pos": (lon0, lat0)}])

        # Satellite subpoint marker
        if SKYFIELD_OK and self.sat_tracker.enabled and self.sat_tracker.loaded:
            ok, slat, slon, _, _ = self.sat_tracker.update(ground_lat_deg=lat0, ground_lon_deg=lon0)
            if ok and slat is not None and slon is not None:
                self.sat_subpoint.setData([{"pos": (float(slon), float(slat))}])
            else:
                self.sat_subpoint.setData([])
        else:
            self.sat_subpoint.setData([])

    # ======= Link View fancy (2.5D + illusion upgrades) =======

    def update_link_view(self, status_color, hpbw_deg_val, snr_db, animated=False):
        self._last_link_view = (status_color, hpbw_deg_val, snr_db)

        bs = (0.0, 0.0)

        # Default
        sat_x = clamp((float(self.sb_lon.value()) / 30.0), -6.0, 6.0)
        sat_y = 15.0
        sat_alt_km = None
        sat_lon = None

        if SKYFIELD_OK and self.sat_tracker.enabled and self.sat_tracker.loaded:
            # ✅ FIX 2: removed the broken extra line, keep only the valid assignment
            ok, _, slon, salt_km, _ = self.sat_tracker.update(
                ground_lat_deg=float(self.sb_lat.value()),
                ground_lon_deg=float(self.sb_lon.value())
            )
            if ok and slon is not None:
                sat_lon = float(slon)
                sat_alt_km = float(salt_km) if salt_km is not None else None

                glon = float(self.sb_lon.value())
                dlon = (sat_lon - glon + 180.0) % 360.0 - 180.0
                sat_x = clamp(dlon / 25.0, -7.5, 7.5)

                if sat_alt_km is not None:
                    sat_y = clamp(11.0 + (sat_alt_km / 2000.0) * 7.5, 9.0, 22.0)

        sat = (sat_x, sat_y)

        # Beam width pulse
        width = clamp(6.0 * (hpbw_deg_val / 5.0), 0.5, 8.0)
        pulse = 1.0 + 0.06 * math.sin(self._anim_phase)
        width_p = width * pulse

        # Size scaling (altitude illusion)
        if sat_alt_km is not None:
            sat_size = clamp(10.0 + (sat_alt_km / 2000.0) * 10.0, 10.0, 24.0)
            glow_size = clamp(28.0 + (sat_alt_km / 2000.0) * 28.0, 28.0, 64.0)
        else:
            sat_size = 16.0
            glow_size = 40.0

        # Shadow: project downwards (illusion)
        shadow = (sat[0] + 0.4, max(0.0, sat[1] - 5.0))

        self.bs_item.setData([{"pos": bs}])
        self.sat_shadow.setData([{"pos": shadow, "size": float(sat_size + 6)}])
        self.sat_item.setData([{"pos": sat, "size": float(sat_size)}])
        self.sat_glow.setData([{"pos": sat, "size": float(glow_size)}])

        pen_main = pg.mkPen(status_color, width=3)
        self.beam_line.setData([bs[0], sat[0]], [bs[1], sat[1]], pen=pen_main)

        cone = np.array([
            [bs[0] - width_p/2, bs[1] + 1.0],
            [bs[0] + width_p/2, bs[1] + 1.0],
            [sat[0], sat[1]],
            [bs[0] - width_p/2, bs[1] + 1.0]
        ], dtype=float)
        self.beam_cone.setData(cone[:, 0], cone[:, 1], pen=pg.mkPen(status_color, width=2))

        fill_color = {"red": (255, 60, 60, 70), "orange": (255, 170, 0, 70), "green": (0, 255, 120, 70)}.get(
            status_color, (0, 255, 120, 70)
        )
        self.beam_fill.setData(cone[:, 0], cone[:, 1], pen=None)
        self.beam_fill.setFillLevel(0)
        self.beam_fill.setBrush(pg.mkBrush(*fill_color))

        # Earth horizon (parabola)
        xs = np.linspace(-10, 10, 200)
        ys = -2.0 + 0.02 * (xs**2)
        self.earth_arc.setData(xs, ys)

        # Dynamic orbit arc (illusion)
        theta = np.linspace(-1.2, 2.6, 160)
        r = 2.0 + 0.25 * math.sin(self._anim_phase * 0.7)
        phase = self._anim_phase * 0.35
        ox = sat[0] + r * np.cos(theta + phase)
        oy = sat[1] + r * np.sin(theta + phase)
        self.orbit_arc.setData(ox, oy)

        # Trail + simple ground track hint
        if SKYFIELD_OK and self.sat_tracker.enabled and self.sat_tracker.loaded:
            self._sat_trail.append((float(sat[0]), float(sat[1])))
            if len(self._sat_trail) >= 2:
                tx = [p[0] for p in self._sat_trail]
                ty = [p[1] for p in self._sat_trail]
                self.trail_item.setData(tx, ty)
            else:
                self.trail_item.setData([], [])

            if sat_lon is not None:
                glon = float(self.sb_lon.value())
                dlon = (sat_lon - glon + 180.0) % 360.0 - 180.0
                gx = clamp(dlon / 25.0, -7.5, 7.5)
                self.groundtrack_item.setData([gx - 0.8, gx + 0.8], [1.0, 1.0])
            else:
                self.groundtrack_item.setData([], [])
        else:
            self.trail_item.setData([], [])
            self.groundtrack_item.setData([], [])

        # Animated packets
        n_packets = 14
        t0 = (self._anim_phase % (2 * math.pi)) / (2 * math.pi)
        ts = (t0 + np.linspace(0, 0.8, n_packets)) % 1.0
        px = bs[0] + ts * (sat[0] - bs[0])
        py = bs[1] + ts * (sat[1] - bs[1])
        spots = [{"pos": (float(px[i]), float(py[i]))} for i in range(n_packets)]
        self.packet_item.setData(spots)

        sat_name = self.sat_tracker.active_name() if (SKYFIELD_OK and self.sat_tracker.enabled and self.sat_tracker.loaded) else ""
        extra = ""
        if sat_name:
            extra += f" | {sat_name}"
        if sat_alt_km is not None:
            extra += f" | ALT={sat_alt_km:.0f} km"

        self.link_text.setText(f"SNR={snr_db:.2f} dB | HPBW={hpbw_deg_val:.3f}°{extra}")
        self.link_text.setColor(status_color)
        self.link_text.setPos(-9, 18)

    # ================== LIVE METRICS ==================

    def push_live_metrics(self, pr_dbm, snr_db, ebn0_db, margin_db):
        now = time.time()
        if self._live_t0 is None:
            self._live_t0 = now
        t = now - self._live_t0

        self._live_x.append(float(t))
        self._live_pr.append(float(pr_dbm))
        self._live_snr.append(float(snr_db))
        self._live_ebn0.append(float(ebn0_db))
        self._live_margin.append(float(margin_db))

        # Update plots
        self.live_curve_pr.setData(list(self._live_x), list(self._live_pr))
        self.live_curve_snr.setData(list(self._live_x), list(self._live_snr))
        self.live_curve_ebn0.setData(list(self._live_x), list(self._live_ebn0))
        self.live_curve_margin.setData(list(self._live_x), list(self._live_margin))

        # keep view following
        if len(self._live_x) >= 2:
            x0 = self._live_x[0]
            x1 = self._live_x[-1]
            self.live_plot_pr.setXRange(x0, x1, padding=0.02)
            self.live_plot_snr.setXRange(x0, x1, padding=0.02)
            self.live_plot_ebn0.setXRange(x0, x1, padding=0.02)

    # ================== UPDATE RESULTS ==================
    # (unchanged from your file)

    def update_results(self):
        f_mhz = float(self.sb_freq.value())
        d_km = float(self.sb_dist.value())

        adv_enabled = self.cb_enable_adv.isChecked()
        if adv_enabled:
            d_km = self.compute_slant_range_km()

        theory_mode = False
        try:
            theory_mode = self.cb_theory_mode.isChecked()
        except Exception:
            theory_mode = False

        bw_hz = float(self.sb_bw_mhz.value()) * 1e6
        rb_hz = float(self.sb_rb_mbps.value()) * 1e6

        ptx_dbm = float(self.sb_tx_pwr_dbm.value())
        gtx = float(self.sb_ue_gain.value())
        grx = float(self.sb_gnb_gain.value())
        ltx = float(self.sb_cable_loss.value())
        lmisc = float(self.sb_misc.value())

        nf = float(self.sb_nf.value())

        D = float(self.sb_dish_d_m.value())
        eta = float(self.sb_eff.value())

        # Protect against near-zero
        f_mhz_safe = max(f_mhz, 1.0)

        hpbw = hpbw_deg(D, f_mhz_safe)
        g_dish = dish_gain_dbi(D, f_mhz_safe, eta)

        mis = float(self.sb_mispoint_deg.value())
        lp = pointing_loss_db(mis, hpbw)

        # Optional: use calculated dish gain as Gtx (keeps UI aligned with parabolic antenna sizing)
        use_dish_gtx = False
        try:
            use_dish_gtx = self.cb_use_dish_as_gtx.isChecked()
        except Exception:
            use_dish_gtx = False

        self.sb_hpbw_ro.setValue(hpbw)
        self.sb_point_loss_ro.setValue(lp)
        self.sb_dish_gain_ro.setValue(g_dish)

        fspl_val = fspl_db(f_mhz_safe, d_km)

        if theory_mode:
            # ===== Satellite-theory budget (matches examples 2.5.6 / 2.5.7) =====
            # Tx side (Earth station)
            ptx_dbw = dbm_to_dbw(ptx_dbm)
            gtx_eff = g_dish if use_dish_gtx else gtx
            eirp_dbw = ptx_dbw + gtx_eff - lp - ltx
            eirp_dbm = dbw_to_dbm(eirp_dbw)

            # Propagation (uplink): FSPL + atmospheric + rain + polarization
            la_db = float(self.sb_theory_atm.value()) + float(self.sb_theory_rain_db.value())
            lpol_db = float(self.sb_theory_pol.value())
            lu_db = fspl_val + la_db + lpol_db

            # Rx side (Satellite): gain from θ3dB and efficiency, then apply coverage edge loss
            gr_max = sat_gain_from_hpbw_dbi(float(self.sb_theory_theta3db.value()), float(self.sb_theory_eta_rx.value()))
            gr = gr_max - float(self.sb_theory_lr.value())

            # System noise temperature referred to antenna input
            tsys_k = rx_system_noise_temp_k(
                float(self.sb_theory_ta.value()),
                float(self.sb_theory_tf.value()),
                float(self.sb_theory_lfrx.value()),
                float(self.sb_theory_frx.value())
            )
            gt_db = gr - 10.0 * math.log10(tsys_k)

            cn0_db_hz = eirp_dbw - lu_db + gt_db + 228.6
            # Equivalent received carrier power at antenna input
            c_dbw = eirp_dbw - lu_db + gr
            pr_dbm = dbw_to_dbm(c_dbw)

            # Noise over B and SNR
            n0_dbw_hz = K_BOLTZ_DBW_HZ_K + 10.0 * math.log10(tsys_k)
            noise_dbw = n0_dbw_hz + 10.0 * math.log10(max(bw_hz, 1.0))
            noise_dbm = dbw_to_dbm(noise_dbw)
            snr_db = pr_dbm - noise_dbm

            ebn0_db = cn0_db_hz - 10.0 * math.log10(max(rb_hz, 1.0))

            targets = {"QPSK": 3.0, "16QAM": 10.0, "64QAM": 16.0, "256QAM": 22.0}
            target = targets.get(self.cb_mod.currentText(), 10.0)
            margin_db = ebn0_db - target

            # Side panel helper labels
            try:
                self.lbl_theory_gt.setText(f"{gt_db:.2f} dB/K")
                self.lbl_theory_cn0.setText(f"{cn0_db_hz:.2f} dBHz")
            except Exception:
                pass

            # For global dock (reuse existing labels)
            wloss = la_db
            total_losses = lu_db

        else:
            # ===== Terrestrial / simplified budget =====
            wloss = self.weather_loss_db(d_km, f_mhz_safe)
            if f_mhz_safe > 24000:
                wloss += 5.0

            total_losses = fspl_val + wloss + lmisc + lp
            if adv_enabled:
                total_losses += float(self.sb_pol_loss.value())
                total_losses += float(self.sb_atm_loss.value())

            eirp_dbm = ptx_dbm + gtx - ltx
            pr_dbm = eirp_dbm + grx - total_losses

            nf = float(self.sb_nf.value())
            noise_dbm = noise_power_dbm(bw_hz, nf)
            snr_db = pr_dbm - noise_dbm

            n0_dbm_hz = -174.0 + nf
            cn0_db_hz = pr_dbm - n0_dbm_hz
            ebn0_db = cn0_db_hz - 10.0 * math.log10(max(rb_hz, 1.0))

            targets = {"QPSK": 3.0, "16QAM": 10.0, "64QAM": 16.0, "256QAM": 22.0}
            target = targets.get(self.cb_mod.currentText(), 10.0)
            margin_db = ebn0_db - target

        mcs = self.suggest_mcs(snr_db)

        if snr_db < 0:
            status = "BAD"
            color = "red"
        elif snr_db < 10:
            status = "MARGINAL"
            color = "orange"
        else:
            status = "GOOD"
            color = "green"

        # Pretty status badge
        self.g_status.setText(status)
        self.g_status.setProperty("status", status)
        self.g_status.style().unpolish(self.g_status)
        self.g_status.style().polish(self.g_status)

        self.g_pr.setText(f"{pr_dbm:.2f}")
        self.g_snr.setText(f"{snr_db:.2f}")
        self.g_mcs.setText(mcs)
        if theory_mode:
            self.g_eirp.setText(f"{dbm_to_dbw(eirp_dbm):.2f} dBW  ({eirp_dbm:.2f} dBm)")
        else:
            self.g_eirp.setText(f"{eirp_dbm:.2f}")
        self.g_cn0.setText(f"{cn0_db_hz:.2f}")
        self.g_ebn0.setText(f"{ebn0_db:.2f}")
        self.g_fspl.setText(f"{fspl_val:.2f}")
        self.g_weather.setText(f"{wloss:.2f}")
        self.g_losses.setText(f"{total_losses:.2f}")
        self.g_margin.setText(f"{margin_db:.2f}")

        ebn0_c = clamp(ebn0_db, -5.0, 25.0)
        self.bar_ebn0.setValue(int((ebn0_c + 5.0) * 100.0 / 30.0))
        snr_c = clamp(snr_db, -10.0, 30.0)
        self.bar_quality.setValue(int((snr_c + 10.0) * 100.0 / 40.0))

        self.g_explain.setText(self.auto_explain(snr_db, d_km, wloss, nf, lp))

        # Satellite-theory validation output (only for theory scenarios)
        try:
            if theory_mode and hasattr(self, 'txt_theory_validation'):
                sname = self.cb_scenario.currentText()
                exp = None
                if sname.startswith('Theory 2.5.6'):
                    exp = {
                        'EIRP_dBW': 71.7,
                        'FSPL_dB': 207.4,
                        'LU_dB': 207.7,
                        'G_T': 6.6,
                        'CN0': 99.2
                    }
                elif sname.startswith('Theory 2.5.7'):
                    exp = {
                        'EIRP_dBW': 71.7,
                        'FSPL_dB': 207.4,
                        'LU_dB': 217.7,
                        'G_T': 6.6,
                        'CN0': 89.2
                    }

                if exp is None:
                    self.txt_theory_validation.setPlainText(
                        'Select a theory scenario (2.5.6 / 2.5.7) for computed vs expected comparison.'
                    )
                else:
                    eirp_dbw = dbm_to_dbw(eirp_dbm)
                    # We stored total_losses = LU in theory mode
                    lu_db = float(total_losses)
                    # G/T recompute quickly
                    gr_max = sat_gain_from_hpbw_dbi(float(self.sb_theory_theta3db.value()), float(self.sb_theory_eta_rx.value()))
                    gr = gr_max - float(self.sb_theory_lr.value())
                    tsys_k = rx_system_noise_temp_k(
                        float(self.sb_theory_ta.value()),
                        float(self.sb_theory_tf.value()),
                        float(self.sb_theory_lfrx.value()),
                        float(self.sb_theory_frx.value())
                    )
                    gt_db = gr - 10.0 * math.log10(tsys_k)

                    def ok_line(name, got, ex, tol=0.2, unit=''):
                        ok = abs(got - ex) <= tol
                        mark = 'OK' if ok else 'CHECK'
                        return f"{name:<10s}: {got:>7.2f}{unit} | exp {ex:>7.2f}{unit} | {mark}"

                    lines = []
                    lines.append('Validation: Theory examples 2.5.6 / 2.5.7')
                    lines.append(ok_line('EIRP', eirp_dbw, exp['EIRP_dBW'], unit=' dBW'))
                    lines.append(ok_line('FSPL', fspl_val, exp['FSPL_dB'], unit=' dB'))
                    lines.append(ok_line('L_u', lu_db, exp['LU_dB'], unit=' dB'))
                    lines.append(ok_line('G/T', gt_db, exp['G_T'], unit=' dB/K'))
                    lines.append(ok_line('C/N0', cn0_db_hz, exp['CN0'], unit=' dBHz'))
                    self.txt_theory_validation.setPlainText('\n'.join(lines))
        except Exception:
            pass

        # Flow cards
        for (card, st) in self.flow_boxes:
            card.setProperty("status", status)
            card.style().unpolish(card)
            card.style().polish(card)
            st.setText(status)

        # Updates
        self.update_waveform(snr_db)
        self.update_constellation(self.cb_mod.currentText(), snr_db)
        self.update_waterfall(eirp_dbm, grx, total_losses, pr_dbm)
        self.update_coverage_map(color)
        self.update_link_view(color, hpbw, snr_db, animated=False)

        # Live metrics
        self.push_live_metrics(pr_dbm, snr_db, ebn0_db, margin_db)

        self._last = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "f_mhz": f_mhz,
            "d_km": d_km,
            "eirp_dbm": eirp_dbm,
            "pr_dbm": pr_dbm,
            "snr_db": snr_db,
            "ebn0_db": ebn0_db,
            "status": status,
        }

    # ================== HISTORY / EXPORT ==================

    def add_snapshot(self):
        if not hasattr(self, "_last"):
            return
        r = self._last
        self.history_rows.append(r)

        row = self.tbl_hist.rowCount()
        self.tbl_hist.insertRow(row)

        cols = [
            r["time"],
            f"{r['f_mhz']:.0f}",
            f"{r['d_km']:.3f}",
            f"{r['eirp_dbm']:.2f}",
            f"{r['pr_dbm']:.2f}",
            f"{r['snr_db']:.2f}",
            f"{r['ebn0_db']:.2f}",
            r["status"],
        ]
        for c, val in enumerate(cols):
            self.tbl_hist.setItem(row, c, QTableWidgetItem(val))

    def export_csv(self):
        if not self.history_rows:
            QMessageBox.information(self, "Export CSV", "No snapshots yet. Click 'Add Snapshot' first.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "uplink_history.csv", "CSV Files (*.csv)")
        if not path:
            return

        keys = ["time", "f_mhz", "d_km", "eirp_dbm", "pr_dbm", "snr_db", "ebn0_db", "status"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for rr in self.history_rows:
                w.writerow({k: rr.get(k) for k in keys})

        QMessageBox.information(self, "Export CSV", f"Saved: {path}")

    # ================== SCENARIOS ==================

    def apply_scenario(self):
        s = self.cb_scenario.currentText()
        if s == "Urban LOS":
            self.sb_dist.setValue(1.5)
            self.sb_misc.setValue(3.0)
            self.sb_freq.setValue(3500)
            self.sb_tx_pwr_dbm.setValue(23.0)
        elif s == "Urban NLOS":
            self.sb_dist.setValue(1.0)
            self.sb_misc.setValue(10.0)
            self.sb_freq.setValue(3500)
        elif s == "Rural LOS":
            self.sb_dist.setValue(5.0)
            self.sb_misc.setValue(2.0)
            self.sb_freq.setValue(900)
        elif s == "Indoor":
            self.sb_dist.setValue(0.2)
            self.sb_misc.setValue(12.0)
            self.sb_freq.setValue(2400)
        elif s == "Satellite-like (LEO demo)":
            self.cb_enable_adv.setChecked(True)
            self.sb_sat_alt_km.setValue(550.0)
            self.sb_elev_deg.setValue(25.0)
            self.sb_freq.setValue(14500)   # MHz (14.5 GHz)
            self.sb_bw_mhz.setValue(36.0)
            self.sb_rb_mbps.setValue(25.0)
            self.sb_misc.setValue(4.0)
        elif s == "Theory 2.5.6 (Uplink clear-sky)":
            # Example 2.5.6 from theory slides
            self.cb_enable_adv.setChecked(False)
            self.cb_theory_mode.setChecked(True)
            self.cb_use_dish_as_gtx.setChecked(True)

            self.sb_freq.setValue(14000.0)
            self.sb_dist.setValue(40000.0)

            # Ptx=100W => 50 dBm (20 dBW)
            self.sb_tx_pwr_dbm.setValue(50.0)
            self.sb_cable_loss.setValue(0.5)

            # Earth station antenna
            self.sb_dish_d_m.setValue(4.0)
            self.sb_eff.setValue(0.60)
            self.sb_mispoint_deg.setValue(0.10)

            # Satellite receiver parameters (from example)
            self.sb_theory_theta3db.setValue(2.0)
            self.sb_theory_eta_rx.setValue(0.55)
            self.sb_theory_lfrx.setValue(1.0)
            self.sb_theory_frx.setValue(3.0)
            self.sb_theory_ta.setValue(290.0)
            self.sb_theory_tf.setValue(290.0)
            self.sb_theory_lr.setValue(3.0)
            self.sb_theory_pol.setValue(0.0)
            self.sb_theory_atm.setValue(0.3)
            self.sb_theory_rain_db.setValue(0.0)

        elif s == "Theory 2.5.7 (Uplink + rain)":
            # Example 2.5.7 from theory slides (adds 10 dB rain attenuation)
            self.cb_enable_adv.setChecked(False)
            self.cb_theory_mode.setChecked(True)
            self.cb_use_dish_as_gtx.setChecked(True)

            self.sb_freq.setValue(14000.0)
            self.sb_dist.setValue(40000.0)

            self.sb_tx_pwr_dbm.setValue(50.0)
            self.sb_cable_loss.setValue(0.5)

            self.sb_dish_d_m.setValue(4.0)
            self.sb_eff.setValue(0.60)
            self.sb_mispoint_deg.setValue(0.10)

            self.sb_theory_theta3db.setValue(2.0)
            self.sb_theory_eta_rx.setValue(0.55)
            self.sb_theory_lfrx.setValue(1.0)
            self.sb_theory_frx.setValue(3.0)
            self.sb_theory_ta.setValue(290.0)
            self.sb_theory_tf.setValue(290.0)
            self.sb_theory_lr.setValue(3.0)
            self.sb_theory_pol.setValue(0.0)
            self.sb_theory_atm.setValue(0.3)
            self.sb_theory_rain_db.setValue(10.0)

        self.update_results()


# ================== APP STYLE (VISUAL UPDATES) ==================

APP_QSS = """
QMainWindow { background: #f5f6f8; }
#NavTitle { font-size: 12pt; font-weight: 800; padding: 10px 12px; color: #1f2430; }
QGroupBox {
    border: 1px solid #dde1ea;
    border-radius: 12px;
    margin-top: 12px;
    background: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 8px;
    color: #1f2430;
    font-weight: 700;
}
QLabel { color: #1f2430; }
#HintLabel { color: #6a7280; }
QComboBox, QDoubleSpinBox, QPlainTextEdit {
    background: #ffffff;
    border: 1px solid #dde1ea;
    border-radius: 10px;
    padding: 6px 10px;
    min-height: 28px;
}
QPushButton {
    background: #ffffff;
    border: 1px solid #dde1ea;
    border-radius: 10px;
    padding: 8px 12px;
    font-weight: 700;
}
QPushButton:hover { background: #f0f2f6; }
QPushButton:pressed { background: #e6eaf2; }

QTabWidget::pane {
    border: 1px solid #dde1ea;
    border-radius: 12px;
    background: #ffffff;
}
QTabBar::tab {
    background: #ffffff;
    border: 1px solid #dde1ea;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    padding: 7px 12px;
    margin-right: 6px;
    color: #4a5160;
    font-weight: 700;
}
QTabBar::tab:selected {
    background: #f0f2f6;
    color: #1f2430;
}

#NavBtn {
    text-align: left;
    padding-left: 12px;
    border-radius: 12px;
    border: 1px solid #dde1ea;
    background: #ffffff;
    color: #1f2430;
    font-weight: 800;
}
#NavBtn:hover { background: #f0f2f6; }
#NavBtn:checked {
    background: #e9edf5;
    border: 1px solid #cfd6e6;
}

#StatusBadge {
    border-radius: 14px;
    padding: 18px;
    font-size: 16pt;
    font-weight: 900;
    color: white;
    background: #2f9e44;
}
#StatusBadge[status="BAD"] { background: #d9480f; }
#StatusBadge[status="MARGINAL"] { background: #f08c00; }
#StatusBadge[status="GOOD"] { background: #2f9e44; }

#ResultLabel { color: #6a7280; font-weight: 700; }
#ResultValue { color: #1f2430; font-weight: 900; }

#FlowCard {
    border: 1px solid #dde1ea;
    border-radius: 14px;
    background: #ffffff;
}
#FlowCard[status="BAD"] { background: #fff4f0; border-color: #ffd5c7; }
#FlowCard[status="MARGINAL"] { background: #fff8e6; border-color: #ffe0a3; }
#FlowCard[status="GOOD"] { background: #ecfff3; border-color: #b7f0c7; }

#FlowTitle { font-weight: 900; color: #1f2430; }
#FlowStatus { font-weight: 900; }
QProgressBar {
    border: 1px solid #dde1ea;
    border-radius: 10px;
    background: #ffffff;
    text-align: center;
    height: 18px;
}
QProgressBar::chunk { border-radius: 10px; background: #339af0; }
"""

# ================== RUN ==================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_QSS)
    win = UplinkApp()
    win.show()
    sys.exit(app.exec_())
