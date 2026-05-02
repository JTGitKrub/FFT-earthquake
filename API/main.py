"""
Seismic Signal Analyzer — RESTful API
======================================
FastAPI backend ที่รับ CSV file และ parameters
แล้วคืน Normalized Acceleration (a/PGA) เป็น JSON

Run locally:
    pip install fastapi uvicorn numpy scipy python-multipart
    uvicorn main:app --reload

Deploy:
    Railway / Render / Fly.io
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, Literal
import numpy as np
from scipy import signal as scipy_signal
import io, json, os

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Seismic Signal Analyzer API",
    description="""
## Seismic Signal Analyzer — RESTful API
**Chiang Mai University**

### Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info |
| `GET` | `/results` | List available pre-computed results |
| `GET` | `/results/{filename}` | Get a specific pre-computed JSON result |
| `POST` | `/analyze` | Upload CSV → get normalized acceleration JSON |
| `POST` | `/fft` | Upload CSV → get FFT spectrum + peaks |
| `GET` | `/health` | Check API status |
    """,
    version="1.0.0",
)

# ── CORS ───────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Results folder ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  DSP Functions
# ══════════════════════════════════════════════════════════════════════════════

def smart_load(content: bytes) -> np.ndarray:
    """Auto-detect delimiter and load numeric data from CSV/TXT/DAT"""
    text = content.decode("utf-8", errors="ignore")
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        raise ValueError("No data found in file")
    delim = "," if "," in lines[0] else None
    raw = np.genfromtxt(
        io.StringIO("\n".join(lines)),
        delimiter=delim,
        invalid_raise=False
    )
    if raw.ndim == 1:
        raw = raw[~np.isnan(raw)].reshape(-1, 1)
    else:
        raw = raw[~np.isnan(raw).all(axis=1)]
    if raw.shape[0] == 0:
        raise ValueError("No valid numeric data found")
    return raw


def compute_fft(data: np.ndarray, fs: float,
                window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT with Hann window"""
    N = len(data)
    wins = {"hann": np.hanning, "hamming": np.hamming,
            "blackman": np.blackman, "rectangular": np.ones}
    win = wins.get(window, np.hanning)(N)
    win_scale = np.sum(win)
    Y = np.fft.fft(data * win)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    h = N // 2
    freqs = freqs[:h]
    amp = (2.0 / win_scale) * np.abs(Y[:h])
    return freqs, amp


def konno_ohmachi_smooth(freqs: np.ndarray,
                          amp: np.ndarray,
                          b: float = 40.0) -> np.ndarray:
    """Konno-Ohmachi smoothing — seismic standard"""
    n = len(freqs)
    smoothed = np.zeros(n)
    for i, fc in enumerate(freqs):
        if fc <= 0:
            smoothed[i] = amp[i]
            continue
        ratio = np.where(freqs / fc <= 0, 1e-10, freqs / fc)
        with np.errstate(divide="ignore", invalid="ignore"):
            arg = b * np.log10(ratio)
            w = np.where(np.abs(arg) < 1e-6, 1.0, (np.sin(arg) / arg) ** 4)
        w[freqs <= 0] = 0
        total = w.sum()
        smoothed[i] = (w * amp).sum() / total if total > 0 else amp[i]
    return smoothed


def build_mask(freqs: np.ndarray,
               hp_on: bool, hp_freq: float,
               lp_on: bool, lp_freq: float,
               taper: float = 0.0) -> np.ndarray:
    """Build HP/LP bandpass filter mask with cosine taper"""
    mask = np.ones(len(freqs))
    if hp_on:
        f_start = max(hp_freq - taper, 0)
        for i, f in enumerate(freqs):
            if f <= f_start:
                mask[i] = 0.0
            elif f < hp_freq:
                mask[i] *= 0.5 * (
                    1 - np.cos(np.pi * (f - f_start) /
                               max(hp_freq - f_start, 1e-10))
                )
    if lp_on:
        f_end = lp_freq + taper
        for i, f in enumerate(freqs):
            if f >= f_end:
                mask[i] = 0.0
            elif f > lp_freq:
                mask[i] *= 0.5 * (
                    1 + np.cos(np.pi * (f - lp_freq) /
                               max(taper, 1e-10))
                )
    return mask


def apply_bandpass_filter(data: np.ndarray,
                           fs: float,
                           hp_freq: float,
                           lp_freq: float,
                           taper: float = 0.0) -> np.ndarray:
    """
    Apply bandpass filter using N-point FFT (matches numpy exactly).
    Uses rfft at N-point resolution to avoid frequency bin mismatch.
    """
    N = len(data)
    Y = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mask = build_mask(freqs, True, hp_freq, True, lp_freq, taper)
    Y_filtered = Y * mask
    return np.fft.irfft(Y_filtered, n=N)


def find_peaks_fft(freqs: np.ndarray,
                    amp: np.ndarray,
                    n: int = 5) -> list[dict]:
    """Find top N dominant peaks in FFT spectrum"""
    if len(amp) == 0:
        return []
    from scipy.signal import find_peaks
    idx, _ = find_peaks(amp, height=np.percentile(amp, 75), distance=5)
    if len(idx) == 0:
        return []
    top = idx[np.argsort(amp[idx])[::-1]][:n]
    return [
        {"freq_hz": round(float(freqs[i]), 4),
         "amplitude": round(float(amp[i]), 5)}
        for i in top
    ]


# ══════════════════════════════════════════════════════════════════════════════
#  Response Schema
# ══════════════════════════════════════════════════════════════════════════════

class DataPoint(BaseModel):
    time_s: float
    norm_accel: float

class AnalyzeResponse(BaseModel):
    description: str
    component: str
    filter: str
    pga: float
    pga_original: float
    fs_hz: float
    dt_s: float
    n_samples: int
    duration_s: float
    top_peaks: list[dict]
    units: dict
    data: list[dict]


# ══════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    """API info and available endpoints"""
    return {
        "message": "Seismic Signal Analyzer API — Chiang Mai University",
        "version": "1.0.0",
        "endpoints": {
            "docs":    "/docs",
            "health":  "/health",
            "results": "/results",
            "analyze": "POST /analyze",
            "fft":     "POST /fft",
        }
    }


@app.get("/results", tags=["Results"])
def list_results():
    """
    ## List Available Pre-computed Results

    Returns a list of all JSON result files stored on the server.
    Use `GET /results/{filename}` to fetch a specific file.
    """
    files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    return {
        "count": len(files),
        "files": files,
        "usage": "GET /results/{filename}"
    }


@app.get("/results/{filename}", tags=["Results"])
def get_result(filename: str):
    """
    ## Get Pre-computed Result

    Fetch a specific pre-computed normalized acceleration JSON file.

    ### Example
    ```
    GET /results/norm_accel_EW.json
    ```
    """
    if not filename.endswith(".json"):
        filename = filename + ".json"

    # Security: prevent path traversal
    filename = os.path.basename(filename)
    filepath = os.path.join(RESULTS_DIR, filename)

    if not os.path.exists(filepath):
        available = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
        raise HTTPException(
            status_code=404,
            detail={
                "error": f"File '{filename}' not found",
                "available": available
            }
        )

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    return JSONResponse(content=data)


@app.post("/results/{filename}", tags=["Results"])
async def save_result(
    filename: str,
    file: UploadFile = File(..., description="JSON result file to store")
):
    """
    ## Save a Result File

    Store a pre-computed JSON result file on the server.
    After saving, it can be retrieved via `GET /results/{filename}`.
    """
    if not filename.endswith(".json"):
        filename = filename + ".json"
    filename = os.path.basename(filename)
    filepath = os.path.join(RESULTS_DIR, filename)

    content = await file.read()
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {
        "message": f"Saved successfully",
        "filename": filename,
        "url": f"/results/{filename}",
        "n_samples": data.get("n_samples"),
        "component": data.get("component"),
        "filter": data.get("filter"),
    }



def health():
    """Check API status"""
    return {"status": "ok", "version": "1.0.0",
            "service": "Seismic Signal Analyzer API"}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="CSV/TXT/DAT seismic signal file"),
    dt: float = Query(0.005, gt=0, description="Time step in seconds (e.g. 0.005 = 200 Hz)"),
    acc_col: int = Query(0, ge=0, description="Acceleration start column (0-based)"),
    component: Literal["EW", "NS", "V", "ACC", "auto"] = Query(
        "auto", description="Component to analyze. 'auto' = first available"
    ),
    hp_freq: float = Query(0.1, gt=0, description="High-pass frequency (Hz)"),
    lp_freq: float = Query(25.0, gt=0, description="Low-pass frequency (Hz)"),
    taper: float = Query(0.0, ge=0, description="Cosine taper width (Hz)"),
    ko_smooth: bool = Query(True, description="Apply Konno-Ohmachi smoothing to FFT"),
    ko_b: float = Query(40.0, gt=0, description="Konno-Ohmachi bandwidth b"),
    save: bool = Query(False, description="Save result to /results/{component}.json for later retrieval"),
):
    """
    ## Analyze Seismic Signal

    Upload a CSV file and get Normalized Acceleration (a/PGA) as JSON.

    ### Parameters
    - **file**: CSV/TXT/DAT file (comma or whitespace delimited)
    - **dt**: Time step in seconds
    - **acc_col**: First acceleration column (0-based)
    - **component**: Which component to analyze
    - **hp_freq / lp_freq**: Bandpass filter frequencies (Hz)
    - **taper**: Cosine taper width (Hz, 0 = brick-wall filter)
    - **ko_smooth**: Apply Konno-Ohmachi smoothing to FFT spectrum
    - **ko_b**: KO smoothing bandwidth (40 = seismic standard)

    ### Returns
    JSON with:
    - Signal metadata (PGA, fs, duration)
    - Top frequency peaks
    - Normalized acceleration data: `[{time_s, norm_accel}]`
    """
    # ── Load file ────────────────────────────────────────────────────────────
    try:
        content = await file.read()
        raw = smart_load(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File parse error: {e}")

    ncols = raw.shape[1]
    fs = 1.0 / dt

    if acc_col >= ncols:
        raise HTTPException(
            status_code=400,
            detail=f"acc_col={acc_col} out of range (file has {ncols} columns)"
        )

    # ── Extract component ─────────────────────────────────────────────────────
    remaining = ncols - acc_col
    if remaining >= 3:
        comp_map = {
            "EW":  raw[:, acc_col].astype(float),
            "NS":  raw[:, acc_col + 1].astype(float),
            "V":   raw[:, acc_col + 2].astype(float),
        }
    else:
        comp_map = {"ACC": raw[:, acc_col].astype(float)}

    if component == "auto":
        comp_name = list(comp_map.keys())[0]
    elif component in comp_map:
        comp_name = component
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Component '{component}' not available. Available: {list(comp_map.keys())}"
        )

    data = comp_map[comp_name]
    N = len(data)
    t = np.arange(N) * dt

    # ── Validate filter params ────────────────────────────────────────────────
    nyq = fs / 2.0
    if hp_freq >= lp_freq:
        raise HTTPException(
            status_code=400,
            detail=f"hp_freq ({hp_freq}) must be less than lp_freq ({lp_freq})"
        )
    if lp_freq >= nyq:
        raise HTTPException(
            status_code=400,
            detail=f"lp_freq ({lp_freq}) must be less than Nyquist ({nyq} Hz)"
        )

    # ── FFT + peaks ───────────────────────────────────────────────────────────
    freqs, amp = compute_fft(data, fs, window="hann")
    if ko_smooth:
        mf = freqs > 0
        amp_s = amp.copy()
        amp_s[mf] = konno_ohmachi_smooth(freqs[mf], amp[mf], b=ko_b)
        amp = amp_s
    mf = freqs > 0
    top_peaks = find_peaks_fft(freqs[mf], amp[mf])

    # ── Apply bandpass filter ─────────────────────────────────────────────────
    data_filt = apply_bandpass_filter(data, fs, hp_freq, lp_freq, taper)

    # ── Normalize ─────────────────────────────────────────────────────────────
    pga_orig = float(np.max(np.abs(data)))
    pga_filt = float(np.max(np.abs(data_filt)))

    if pga_filt == 0:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Filter produced zero signal. "
                f"Try adjusting hp_freq/lp_freq. "
                f"Current: HP={hp_freq}, LP={lp_freq} Hz. "
                f"Available peaks: {top_peaks[:3]}"
            )
        )

    norm = data_filt / pga_filt

    # ── Build filter label ────────────────────────────────────────────────────
    filter_label = f"HP>{hp_freq:.3f}Hz + LP<{lp_freq:.3f}Hz"

    # ── Return response ───────────────────────────────────────────────────────
    result = {
        "description": (
            f"Normalized Acceleration (a/PGA) | "
            f"Component: {comp_name} | Filter: {filter_label}"
        ),
        "component":    comp_name,
        "filter":       filter_label,
        "pga":          round(pga_filt, 8),
        "pga_original": round(pga_orig, 8),
        "fs_hz":        fs,
        "dt_s":         dt,
        "n_samples":    N,
        "duration_s":   round(float(t[-1]), 4),
        "top_peaks":    top_peaks,
        "units": {
            "time_s":     "seconds",
            "norm_accel": "dimensionless (a/PGA) range: ±1",
            "pga":        "same unit as input acceleration",
        },
        "data": [
            {
                "time_s":     round(float(t[i]), 6),
                "norm_accel": round(float(norm[i]), 8),
            }
            for i in range(N)
        ],
    }

    # ── Auto-save if requested ────────────────────────────────────────────────
    if save:
        save_filename = f"norm_accel_{comp_name}.json"
        save_path = os.path.join(RESULTS_DIR, save_filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        result["saved_as"] = f"/results/{save_filename}"

    return result


@app.post("/fft", tags=["Analysis"])
async def get_fft(
    file: UploadFile = File(...),
    dt: float = Query(0.005, gt=0),
    acc_col: int = Query(0, ge=0),
    ko_smooth: bool = Query(True),
    ko_b: float = Query(40.0),
):
    """
    ## Get FFT Spectrum Only

    Returns FFT spectrum with detected peaks (without filtering).
    Useful for exploring the signal before choosing filter frequencies.
    """
    try:
        content = await file.read()
        raw = smart_load(content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ncols = raw.shape[1]
    fs = 1.0 / dt
    if acc_col >= ncols:
        raise HTTPException(status_code=400,
                            detail=f"acc_col out of range (max {ncols-1})")

    remaining = ncols - acc_col
    if remaining >= 3:
        comp_map = {
            "EW": raw[:, acc_col].astype(float),
            "NS": raw[:, acc_col + 1].astype(float),
            "V":  raw[:, acc_col + 2].astype(float),
        }
    else:
        comp_map = {"ACC": raw[:, acc_col].astype(float)}

    result = {}
    for comp, data in comp_map.items():
        freqs, amp = compute_fft(data, fs)
        if ko_smooth:
            mf = freqs > 0
            amp_s = amp.copy()
            amp_s[mf] = konno_ohmachi_smooth(freqs[mf], amp[mf], b=ko_b)
            amp = amp_s
        mf = freqs > 0
        result[comp] = {
            "peaks": find_peaks_fft(freqs[mf], amp[mf]),
            "spectrum": [
                {"freq_hz": round(float(f), 5),
                 "amplitude": round(float(a), 6)}
                for f, a in zip(freqs[mf], amp[mf])
            ]
        }

    return {
        "fs_hz": fs, "dt_s": dt, "n_samples": len(raw),
        "duration_s": round(len(raw) * dt, 4),
        "ko_smoothing": {"enabled": ko_smooth, "b": ko_b},
        "components": result
    }
