"""
Seismic Signal Analyzer — Patrick's Earthquake Engineering Lab
Streamlit Web Application
"""

import streamlit as st
import numpy as np
from scipy import signal as scipy_signal
from scipy.integrate import cumulative_trapezoid
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import json
import io

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Seismic Signal Analyzer",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Source Serif 4', Georgia, serif;
    }
    .stApp {
        background-color: #FAFAFA;
    }
    .main-header {
        background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(21,101,192,0.25);
    }
    .main-header h1 {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
        opacity: 0.85;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-card {
        background: white;
        border: 1.5px solid #E0E0E0;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .metric-label {
        font-size: 0.75rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1565C0;
        font-family: 'JetBrains Mono', monospace;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #1565C0;
        border-bottom: 2px solid #1565C0;
        padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
    .info-box {
        background: #E3F2FD;
        border-left: 4px solid #1565C0;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #1A237E;
        margin: 0.5rem 0;
    }
    .success-box {
        background: #E8F5E9;
        border-left: 4px solid #2E7D32;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #1B5E20;
        margin: 0.5rem 0;
    }
    .warning-box {
        background: #FFF3E0;
        border-left: 4px solid #E65100;
        border-radius: 4px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
        color: #BF360C;
        margin: 0.5rem 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #F5F5F5;
        border-right: 1.5px solid #E0E0E0;
    }
    .stButton > button {
        font-family: 'Source Serif 4', serif;
        font-weight: 600;
        border-radius: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'Source Serif 4', serif;
        font-size: 0.95rem;
        font-weight: 600;
    }
    code, pre {
        font-family: 'JetBrains Mono', monospace;
    }
</style>
""", unsafe_allow_html=True)

# ── DSP Functions ─────────────────────────────────────────────────────────────

def smart_load(file_bytes, filename):
    """Auto-detect delimiter and load numeric data"""
    try:
        text = file_bytes.decode('utf-8', errors='ignore')
        lines = [l.strip() for l in text.splitlines()
                 if l.strip() and not l.strip().startswith('#')]
        delim = ',' if ',' in lines[0] else None
        raw = np.genfromtxt(
            io.StringIO('\n'.join(lines)),
            delimiter=delim, invalid_raise=False)
        if raw.ndim == 1:
            raw = raw[~np.isnan(raw)].reshape(-1, 1)
        else:
            raw = raw[~np.isnan(raw).all(axis=1)]
        return raw
    except Exception as e:
        st.error(f"Load error: {e}")
        return None


def compute_fft(data, fs, window="hann", ymode="Linear", one_sided=True):
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
    if ymode == "dB":
        amp = 20 * np.log10(np.maximum(amp, 1e-12))
    elif ymode == "PSD":
        amp = (amp ** 2) / (fs * win_scale)
        amp = np.maximum(amp, 1e-30)
    return freqs, amp


def konno_ohmachi_smooth(freqs, amp, b=40.0):
    n = len(freqs)
    smoothed = np.zeros(n)
    for i, fc in enumerate(freqs):
        if fc <= 0:
            smoothed[i] = amp[i]; continue
        ratio = np.where(freqs / fc <= 0, 1e-10, freqs / fc)
        with np.errstate(divide='ignore', invalid='ignore'):
            arg = b * np.log10(ratio)
            w = np.where(np.abs(arg) < 1e-6, 1.0, (np.sin(arg)/arg)**4)
        w[freqs <= 0] = 0
        total = w.sum()
        smoothed[i] = (w * amp).sum() / total if total > 0 else amp[i]
    return smoothed


def build_mask(freqs, hp_on, hp_freq, lp_on, lp_freq, taper=0.0):
    mask = np.ones(len(freqs))
    if hp_on:
        f_start = max(hp_freq - taper, 0)
        for i, f in enumerate(freqs):
            if f <= f_start: mask[i] = 0.0
            elif f < hp_freq:
                mask[i] *= 0.5*(1 - np.cos(
                    np.pi*(f-f_start)/max(hp_freq-f_start, 1e-10)))
    if lp_on:
        f_end = lp_freq + taper
        for i, f in enumerate(freqs):
            if f >= f_end: mask[i] = 0.0
            elif f > lp_freq:
                mask[i] *= 0.5*(1 + np.cos(
                    np.pi*(f-lp_freq)/max(taper, 1e-10)))
    return mask


def plotly_theme():
    return dict(
        plot_bgcolor='white',
        paper_bgcolor='#F5F5F5',
        font=dict(family='Georgia, serif', size=12, color='#1A1A1A'),
        xaxis=dict(showgrid=True, gridcolor='#EEEEEE',
                   linecolor='#BDBDBD', mirror=True),
        yaxis=dict(showgrid=True, gridcolor='#EEEEEE',
                   linecolor='#BDBDBD', mirror=True),
        margin=dict(l=60, r=30, t=50, b=50),
    )

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "EW":  "#1565C0",
    "NS":  "#2E7D32",
    "V":   "#6A1B9A",
    "ACC": "#1565C0",
}

# ═════════════════════════════════════════════════════════════════════════════
#  Header
# ═════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <h1>🌍 Seismic Signal Analyzer</h1>
    <p>Patrick's Earthquake Engineering Lab  |  FFT · Bandpass Filter · Acc/Vel/Disp</p>
</div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Sidebar — File Upload & Config
# ═════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### 📂 Load Signal File")
    uploaded = st.file_uploader(
        "Upload CSV / TXT / DAT",
        type=["csv", "txt", "dat", "out"],
        help="รองรับ delimiter: comma หรือ whitespace")

    if uploaded:
        raw = smart_load(uploaded.read(), uploaded.name)
        if raw is not None:
            st.session_state['raw'] = raw
            ncols = raw.shape[1]
            nrows = raw.shape[0]
            st.markdown(f"""
            <div class="success-box">
            ✅ <b>{uploaded.name}</b><br>
            {nrows:,} samples  |  {ncols} columns
            </div>""", unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### ⚙️ Signal Configuration")

            dt = st.number_input(
                "Time step dt (seconds)",
                min_value=0.0001, max_value=10.0,
                value=0.005, format="%.4f",
                help="dt=0.005 → fs=200 Hz")
            fs = 1.0 / dt
            st.caption(f"→ Sampling rate: **{fs:.2f} Hz**")

            acc_col = st.number_input(
                "Acceleration start column (0-based)",
                min_value=0, max_value=ncols-1, value=0)

            remaining = ncols - acc_col
            if remaining >= 3:
                mode = "3comp"
                st.markdown("""<div class="info-box">
                    🔵 3-Component mode: EW / NS / V</div>""",
                    unsafe_allow_html=True)
            else:
                mode = "single"
                st.markdown("""<div class="info-box">
                    🔵 Single component mode: ACC</div>""",
                    unsafe_allow_html=True)

            st.session_state['config'] = {
                'dt': dt, 'fs': fs,
                'acc_col': int(acc_col), 'mode': mode
            }

# ── Guard ─────────────────────────────────────────────────────────────────────
if 'raw' not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding:5rem 2rem; color:#999;">
        <div style="font-size:4rem;">📁</div>
        <div style="font-size:1.3rem; font-weight:600; margin-top:1rem;">
            Upload a signal file to begin
        </div>
        <div style="font-size:0.95rem; margin-top:0.5rem;">
            Supports CSV, TXT, DAT — comma or whitespace delimited
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Extract components ────────────────────────────────────────────────────────
raw    = st.session_state['raw']
cfg    = st.session_state['config']
dt     = cfg['dt']; fs = cfg['fs']; acc_col = cfg['acc_col']
mode   = cfg['mode']
N      = raw.shape[0]
t      = np.arange(N) * dt

if mode == "3comp":
    comp_data = {
        "EW": raw[:, acc_col].astype(float),
        "NS": raw[:, acc_col+1].astype(float),
        "V":  raw[:, acc_col+2].astype(float),
    }
else:
    comp_data = {"ACC": raw[:, acc_col].astype(float)}

# ── Metric cards ──────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Samples</div>
        <div class="metric-value">{N:,}</div></div>""",
        unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Sampling Rate</div>
        <div class="metric-value">{fs:.1f} Hz</div></div>""",
        unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Duration</div>
        <div class="metric-value">{N*dt:.2f} s</div></div>""",
        unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="metric-card">
        <div class="metric-label">Mode</div>
        <div class="metric-value">{"3-Comp" if mode=="3comp" else "Single"}</div></div>""",
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
#  Tabs
# ═════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊  FFT + Filter",
    "📐  Acc / Vel / Disp",
    "📄  About",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — FFT + Filter
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    col_ctrl, col_plot = st.columns([1, 2.8])

    with col_ctrl:
        st.markdown('<div class="section-header">⚙️ FFT Settings</div>',
                    unsafe_allow_html=True)
        window   = st.selectbox("Window", ["hann","hamming","blackman","rectangular"])
        ymode    = st.selectbox("Y-axis", ["Linear","dB","PSD"])
        log_x    = st.checkbox("Log X-axis", value=True)
        one_side = st.checkbox("One-sided Spectrum", value=True)
        show_pk  = st.checkbox("Show Peaks", value=True)

        st.markdown('<div class="section-header">〰️ Konno-Ohmachi</div>',
                    unsafe_allow_html=True)
        use_ko  = st.checkbox("Enable Smoothing", value=True)
        b_val   = st.slider("Bandwidth b", 5, 200, 40)

        st.markdown('<div class="section-header">🔁 Bandpass Filter</div>',
                    unsafe_allow_html=True)
        hp_on   = st.checkbox("High-Pass (HP)", value=True)
        hp_freq = st.number_input("HP freq (Hz)", 0.001, 1000.0, 0.1, format="%.3f")
        lp_on   = st.checkbox("Low-Pass (LP)", value=True)
        lp_freq = st.number_input("LP freq (Hz)", 0.001, 10000.0, 25.0, format="%.3f")
        taper   = st.number_input("Taper (Hz)", 0.0, 100.0, 0.0, format="%.3f")

        run_fft    = st.button("⚡ Compute FFT", use_container_width=True, type="primary")
        run_filter = st.button("🔁 Apply Filter + Convert", use_container_width=True)

    with col_plot:
        # ── Time series ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Time Series</div>',
                    unsafe_allow_html=True)
        fig_time = make_subplots(
            rows=len(comp_data), cols=1,
            shared_xaxes=True,
            subplot_titles=list(comp_data.keys()))
        for i, (comp, data) in enumerate(comp_data.items(), 1):
            fig_time.add_trace(
                go.Scatter(x=t, y=data, mode='lines',
                           line=dict(color=COLORS.get(comp,"#1565C0"), width=0.8),
                           name=comp),
                row=i, col=1)
        fig_time.update_layout(
            height=120*len(comp_data)+100,
            showlegend=False,
            **plotly_theme())
        fig_time.update_xaxes(title_text="Time (s)", row=len(comp_data), col=1)
        st.plotly_chart(fig_time, use_container_width=True)

        # ── FFT ──────────────────────────────────────────────────────────
        if run_fft:
            st.markdown('<div class="section-header">📊 FFT Spectrum</div>',
                        unsafe_allow_html=True)
            ylabel_map = {"Linear":"Amplitude","dB":"Magnitude (dB)","PSD":"PSD (units²/Hz)"}
            fig_fft = make_subplots(
                rows=len(comp_data), cols=1,
                shared_xaxes=True,
                subplot_titles=[f"FFT — {c}" for c in comp_data])

            stats_rows = []
            for i, (comp, data) in enumerate(comp_data.items(), 1):
                color = COLORS.get(comp,"#1565C0")
                freqs, amp = compute_fft(data, fs, window=window,
                                         ymode=ymode, one_sided=one_side)
                if use_ko and len(freqs) > 0:
                    msk = freqs > 0
                    amp_s = amp.copy()
                    amp_s[msk] = konno_ohmachi_smooth(freqs[msk], amp[msk], b=b_val)
                    amp = amp_s

                mf = freqs > 0
                fig_fft.add_trace(
                    go.Scatter(x=freqs[mf], y=amp[mf], mode='lines',
                               line=dict(color=color, width=1.2),
                               fill='tozeroy', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:],16)},0.1)',
                               name=comp),
                    row=i, col=1)

                if show_pk:
                    pk_idx, _ = find_peaks(amp[mf], height=np.percentile(amp[mf],75), distance=5)
                    top = pk_idx[np.argsort(amp[mf][pk_idx])[::-1]][:3]
                    for p in top:
                        fig_fft.add_vline(
                            x=freqs[mf][p], line_dash="dash",
                            line_color=color, opacity=0.5,
                            annotation_text=f"{freqs[mf][p]:.3f}Hz",
                            annotation_font_size=10, row=i, col=1)
                        stats_rows.append({
                            "Component": comp,
                            "Peak Freq (Hz)": f"{freqs[mf][p]:.4f}",
                            "Amplitude": f"{amp[mf][p]:.5f}"
                        })

            fig_fft.update_layout(
                height=200*len(comp_data)+100,
                showlegend=False,
                **plotly_theme())
            if log_x:
                for i in range(1, len(comp_data)+1):
                    fig_fft.update_xaxes(type='log', row=i, col=1)
            fig_fft.update_xaxes(title_text="Frequency (Hz)",
                                  row=len(comp_data), col=1)
            st.plotly_chart(fig_fft, use_container_width=True)

            if stats_rows:
                st.markdown("**🔍 Detected Peaks**")
                st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

            # Save button
            st.session_state['fft_done'] = True

        # ── Filter + Normalized ───────────────────────────────────────────
        if run_filter:
            st.markdown('<div class="section-header">🔁 Filter Results</div>',
                        unsafe_allow_html=True)

            json_exports = {}

            for comp, data in comp_data.items():
                color = COLORS.get(comp, "#1565C0")
                N_d = len(data)
                Y = np.fft.rfft(data)
                freqs_r = np.fft.rfftfreq(N_d, d=1/fs)
                mask = build_mask(freqs_r, hp_on, hp_freq, lp_on, lp_freq, taper)
                data_filt = np.fft.irfft(Y * mask, n=N_d)

                pga_o = np.max(np.abs(data))
                pga_f = np.max(np.abs(data_filt))
                norm  = data_filt / pga_f if pga_f > 0 else data_filt

                # Build filter label
                parts = []
                if hp_on: parts.append(f"HP>{hp_freq:.3f}Hz")
                if lp_on: parts.append(f"LP<{lp_freq:.3f}Hz")
                flabel = " + ".join(parts)

                # 3-subplot figure
                fig = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=[
                        f"{comp} — Original vs Modified",
                        f"Modified Waveform  ({flabel})",
                        f"Normalized Acceleration  (a / PGA)  [{flabel}]"
                    ],
                    vertical_spacing=0.1)

                # Subplot 1
                fig.add_trace(go.Scatter(
                    x=t, y=data, mode='lines',
                    line=dict(color='#AAAAAA', width=0.8),
                    name='Original', opacity=0.7), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=t, y=data_filt, mode='lines',
                    line=dict(color=color, width=1.3),
                    name='Modified'), row=1, col=1)

                # Subplot 2
                fig.add_trace(go.Scatter(
                    x=t, y=data_filt, mode='lines',
                    line=dict(color='#2E7D32', width=1.3),
                    fill='tozeroy',
                    fillcolor='rgba(46,125,50,0.1)',
                    name='Modified'), row=2, col=1)

                # Subplot 3 — Normalized ±1
                fig.add_trace(go.Scatter(
                    x=t, y=norm, mode='lines',
                    line=dict(color='#1565C0', width=1.3),
                    fill='tozeroy',
                    fillcolor='rgba(21,101,192,0.08)',
                    name='a/PGA'), row=3, col=1)
                # +1 line
                fig.add_hline(y= 1.0, line_dash="dash", line_color="#C62828",
                              line_width=1.5, row=3, col=1,
                              annotation_text=f"+1 (PGA={pga_f:.5f})",
                              annotation_font_size=10)
                fig.add_hline(y=-1.0, line_dash="dash", line_color="#C62828",
                              line_width=1.5, row=3, col=1,
                              annotation_text=f"−1",
                              annotation_font_size=10)
                fig.add_hline(y=0.0, line_color="#888888",
                              line_width=0.8, opacity=0.5, row=3, col=1)

                # Peak markers on normalized
                min_dist = max(int(fs * 0.3), 1)
                pk_pos = find_peaks( norm, height=0.5, distance=min_dist)[0]
                pk_neg = find_peaks(-norm, height=0.5, distance=min_dist)[0]
                for pk in pk_pos:
                    fig.add_scatter(x=[t[pk]], y=[norm[pk]], mode='markers',
                                    marker=dict(color='#1565C0', size=6),
                                    row=3, col=1, showlegend=False)
                for pk in pk_neg:
                    fig.add_scatter(x=[t[pk]], y=[norm[pk]], mode='markers',
                                    marker=dict(color='#C62828', size=6),
                                    row=3, col=1, showlegend=False)

                fig.update_yaxes(range=[-1.45, 1.45],
                                 tickvals=[-1, -0.5, 0, 0.5, 1],
                                 ticktext=["-1.0","-0.5","0.0","+0.5","+1.0"],
                                 row=3, col=1)

                fig.update_layout(
                    height=750, showlegend=False,
                    title=dict(
                        text=f"<b>Component: {comp}</b>  |  "
                             f"PGA: {pga_o:.5f} → {pga_f:.5f}  |  "
                             f"RMS: {np.sqrt(np.mean(data**2)):.5f} → "
                             f"{np.sqrt(np.mean(data_filt**2)):.5f}",
                        font=dict(size=12, color='#555555')),
                    **plotly_theme())
                fig.update_xaxes(title_text="Time (s)", row=3, col=1)
                st.plotly_chart(fig, use_container_width=True)

                # Store for JSON export
                json_exports[comp] = {
                    "t": t, "norm": norm,
                    "pga_f": pga_f, "flabel": flabel,
                    "parts": parts
                }

            # ── JSON Export ───────────────────────────────────────────────
            st.markdown('<div class="section-header">📄 Export JSON</div>',
                        unsafe_allow_html=True)
            exp_comp = st.selectbox(
                "Select component to export",
                list(json_exports.keys()))

            if exp_comp and st.button("📄 Generate JSON", type="primary"):
                ex = json_exports[exp_comp]
                payload = {
                    "description": (
                        f"Normalized Acceleration (a/PGA)  |  "
                        f"Component: {exp_comp}  |  "
                        f"Filter: {ex['flabel']}"
                    ),
                    "component":  exp_comp,
                    "filter":     ex['flabel'],
                    "pga":        float(ex['pga_f']),
                    "fs_hz":      float(fs),
                    "dt_s":       float(dt),
                    "n_samples":  int(len(ex['norm'])),
                    "duration_s": float(ex['t'][-1]),
                    "units": {
                        "time_s":     "seconds",
                        "norm_accel": "dimensionless (a/PGA) range: ±1",
                        "pga":        "same unit as input"
                    },
                    "data": [
                        {"time_s": round(float(tt), 6),
                         "norm_accel": round(float(aa), 8)}
                        for tt, aa in zip(ex['t'], ex['norm'])
                    ]
                }
                json_str = json.dumps(payload, indent=2, ensure_ascii=False)
                st.download_button(
                    label=f"⬇️ Download norm_accel_{exp_comp}.json",
                    data=json_str,
                    file_name=f"norm_accel_{exp_comp}.json",
                    mime="application/json",
                    use_container_width=True)
                st.markdown(f"""<div class="success-box">
                    ✅ Ready to download!<br>
                    Records: <b>{len(payload['data']):,}</b>  |
                    PGA: <b>{ex['pga_f']:.6f}</b>  |
                    Filter: <b>{ex['flabel']}</b>
                </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — Acc / Vel / Disp
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_c2, col_p2 = st.columns([1, 2.8])

    with col_c2:
        st.markdown('<div class="section-header">📌 Settings</div>',
                    unsafe_allow_html=True)
        input_type = st.selectbox(
            "Input Signal Type",
            ["Acceleration (m/s²)", "Velocity (m/s)", "Displacement (m)"])

        unit_opt = st.selectbox("Unit Conversion Factor", [
            "1.0  (no conversion)",
            "9.81  (g → m/s²)",
            "0.01  (cm/s² → m/s²)",
            "0.001  (mm/s² → m/s²)",
            "Custom"])
        factor_map = {
            "1.0  (no conversion)": 1.0,
            "9.81  (g → m/s²)": 9.81,
            "0.01  (cm/s² → m/s²)": 0.01,
            "0.001  (mm/s² → m/s²)": 0.001,
        }
        if unit_opt == "Custom":
            factor = st.number_input("Custom factor", value=1.0, format="%.4f")
        else:
            factor = factor_map[unit_opt]
            st.caption(f"Factor = **{factor}**")

        do_detrend = st.checkbox("Detrend", value=True)
        do_demean  = st.checkbox("Demean", value=True)

        comp_sel = st.selectbox(
            "Component",
            ["All"] + list(comp_data.keys()))

        run_avd = st.button("⚡ Convert", use_container_width=True, type="primary")

        st.markdown('<div class="section-header">💾 Export</div>',
                    unsafe_allow_html=True)
        export_type = st.selectbox("Export", ["Acceleration","Velocity","Displacement"])

    with col_p2:
        if run_avd:
            comps_run = (comp_data if comp_sel == "All"
                         else {comp_sel: comp_data[comp_sel]})
            idx = ["Acceleration (m/s²)","Velocity (m/s)","Displacement (m)"].index(input_type)
            results = {}
            stats_avd = []

            for comp, sig in comps_run.items():
                s = sig.copy() * factor
                if do_demean:  s -= np.mean(s)
                if do_detrend: s = scipy_signal.detrend(s)

                if idx == 0:
                    acc = s
                    vel  = cumulative_trapezoid(acc, dx=dt, initial=0)
                    disp = cumulative_trapezoid(vel, dx=dt, initial=0)
                elif idx == 1:
                    vel = s
                    disp = cumulative_trapezoid(vel, dx=dt, initial=0)
                    acc  = np.gradient(vel, dt)
                else:
                    disp = s
                    vel  = np.gradient(disp, dt)
                    acc  = np.gradient(vel, dt)

                results[comp] = {"acc": acc, "vel": vel, "disp": disp}
                stats_avd.append({
                    "Component": comp,
                    "PGA (m/s²)": f"{np.max(np.abs(acc)):.6f}",
                    "PGV (m/s)":  f"{np.max(np.abs(vel)):.6f}",
                    "PGD (m)":    f"{np.max(np.abs(disp)):.6f}",
                })

            # Stats table
            st.markdown("**📊 Peak Ground Motion**")
            st.dataframe(pd.DataFrame(stats_avd), use_container_width=True)

            # Plots
            sig_types = [
                ("Acceleration (m/s²)", "acc", "#1565C0"),
                ("Velocity (m/s)",      "vel", "#2E7D32"),
                ("Displacement (m)",    "disp","#6A1B9A"),
            ]
            for comp, res in results.items():
                fig_avd = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    subplot_titles=[f"{comp} — {s[0]}" for s in sig_types])
                for i, (title, key, c) in enumerate(sig_types, 1):
                    fig_avd.add_trace(
                        go.Scatter(x=t, y=res[key], mode='lines',
                                   line=dict(color=c, width=0.9),
                                   name=title),
                        row=i, col=1)
                fig_avd.update_layout(
                    height=650, showlegend=False,
                    title=dict(text=f"<b>Component: {comp}</b>",
                               font=dict(size=13)),
                    **plotly_theme())
                fig_avd.update_xaxes(title_text="Time (s)", row=3, col=1)
                st.plotly_chart(fig_avd, use_container_width=True)

            # Export CSV
            key_map = {"Acceleration":"acc","Velocity":"vel","Displacement":"disp"}
            k = key_map[export_type]
            cols_data = [t]
            col_names = ["Time_s"]
            for comp, res in results.items():
                cols_data.append(res[k]); col_names.append(f"{comp}_{k}")
            Nm = min(len(c) for c in cols_data)
            df_exp = pd.DataFrame(
                np.column_stack([c[:Nm] for c in cols_data]),
                columns=col_names)
            csv_bytes = df_exp.to_csv(index=False).encode()
            st.download_button(
                label=f"⬇️ Download {export_type}.csv",
                data=csv_bytes,
                file_name=f"{export_type.lower()}.csv",
                mime="text/csv",
                use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — About
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    ## 🌍 Seismic Signal Analyzer
    **Patrick's Earthquake Engineering Lab**

    ---

    ### Features
    | Feature | Description |
    |---------|-------------|
    | **FFT Analysis** | Fast Fourier Transform with multiple window functions |
    | **Konno-Ohmachi** | Standard seismic smoothing algorithm |
    | **Bandpass Filter** | HP/LP filter via FFT masking with cosine taper |
    | **Normalized Plot** | a/PGA between ±1 with peak markers |
    | **JSON Export** | Normalized acceleration with time for further analysis |
    | **Acc/Vel/Disp** | Integration from acceleration to velocity and displacement |

    ---

    ### How to Use
    1. **Upload** a CSV or TXT file in the sidebar
    2. **Configure** dt and column settings
    3. Go to **FFT + Filter** tab
    4. Set FFT settings and click **Compute FFT**
    5. Set HP/LP frequencies and click **Apply Filter + Convert**
    6. **Download JSON** of normalized acceleration

    ---

    ### Input File Format
    | Columns | Interpretation |
    |---------|---------------|
    | 1 column | `[ACC]` |
    | 2 columns | `[Time, ACC]` |
    | 3 columns | `[EW, NS, V]` |
    | 4+ columns | `[Time, EW, NS, V, ...]` |

    ---
    *Built with Streamlit · NumPy · SciPy · Plotly*
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#999; font-size:0.85rem; font-family:Georgia,serif;'>"
    "Seismic Signal Analyzer  ·  Patrick's Earthquake Engineering Lab  ·  "
    "Built with Streamlit</p>",
    unsafe_allow_html=True)