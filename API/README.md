# Seismic Signal Analyzer — REST API

**Chiang Mai University**  
FastAPI backend ที่รับ CSV seismic signal แล้วคืน Normalized Acceleration (a/PGA) เป็น JSON

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | ตรวจสอบ API status |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc documentation |
| `POST` | `/analyze` | วิเคราะห์ signal → normalized JSON |
| `POST` | `/fft` | ดู FFT spectrum + peaks เท่านั้น |

---

## ติดตั้งและรัน

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

เปิดที่ http://localhost:8000/docs จะเห็น Swagger UI

---

## ตัวอย่างการเรียก API

### 1. ดู FFT peaks ก่อน (เลือก filter frequency)
```bash
curl -X POST http://localhost:8000/fft \
  -F "file=@mex_test_v2.csv" \
  -F "dt=0.005"
```

### 2. วิเคราะห์ + Filter + Normalize
```bash
curl -X POST "http://localhost:8000/analyze?dt=0.005&hp_freq=1.081&lp_freq=1.083" \
  -F "file=@mex_test_v2.csv"
```

### Python Client
```python
import requests

url = "http://localhost:8000/analyze"

with open("mex_test_v2.csv", "rb") as f:
    response = requests.post(url, 
        files={"file": f},
        params={
            "dt": 0.005,
            "hp_freq": 1.081,
            "lp_freq": 1.083,
            "component": "EW",
        }
    )

data = response.json()
print(f"PGA: {data['pga']}")
print(f"Peaks: {data['top_peaks']}")
print(f"Samples: {data['n_samples']}")
```

### JavaScript/Fetch Client
```javascript
const formData = new FormData();
formData.append('file', csvFile);

const response = await fetch(
  'http://localhost:8000/analyze?dt=0.005&hp_freq=1.081&lp_freq=1.083',
  { method: 'POST', body: formData }
);

const result = await response.json();
console.log(result.data[0]); // {time_s: 0, norm_accel: -0.943}
```

---

## Response Format

```json
{
  "description": "Normalized Acceleration (a/PGA) | Component: EW | Filter: HP>1.081Hz + LP<1.083Hz",
  "component": "EW",
  "filter": "HP>1.081Hz + LP<1.083Hz",
  "pga": 1.943165,
  "pga_original": 49.77,
  "fs_hz": 200.0,
  "dt_s": 0.005,
  "n_samples": 5358,
  "duration_s": 26.785,
  "top_peaks": [
    {"freq_hz": 1.082, "amplitude": 3.205}
  ],
  "units": {
    "time_s": "seconds",
    "norm_accel": "dimensionless (a/PGA) range: ±1"
  },
  "data": [
    {"time_s": 0.0, "norm_accel": -0.94316051},
    {"time_s": 0.005, "norm_accel": -0.93131537}
  ]
}
```

---

## Deploy ฟรีบน Railway

1. Push ไฟล์ `main.py` และ `requirements.txt` ขึ้น GitHub
2. ไปที่ **railway.app** → New Project → Deploy from GitHub
3. Railway detect FastAPI อัตโนมัติ
4. ได้ URL เช่น `https://seismic-api.railway.app`

### หรือ Deploy บน Render

1. ไปที่ **render.com** → New Web Service
2. Connect GitHub repo
3. Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
4. ได้ URL เช่น `https://seismic-api.onrender.com`
