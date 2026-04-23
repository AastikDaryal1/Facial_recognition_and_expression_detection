# Face & Emotion Detection — Production System

Converts the original Google Colab notebook into a fully structured,
production-ready VS Code project with separated pipelines, proper logging,
a REST API, Docker support, and unit tests.

---

## Project Structure

```
face_emotion_system/
│
├── config/
│   └── settings.py            ← ALL config values (env-driven, no hard-coding)
│
├── pipelines/                 ← One file per pipeline stage
│   ├── data_ingestion.py      ← PIPELINE 1: Download from GCS
│   ├── preprocessing.py       ← PIPELINE 2: Clean + validate images
│   ├── feature_extraction.py  ← PIPELINE 3: FaceNet embeddings + cache
│   ├── training.py            ← PIPELINE 4: SVM train + evaluate + save
│   ├── inference.py           ← PIPELINE 5: Static image prediction
│   ├── realtime.py            ← PIPELINE 6: Live webcam loop
│   └── deployment.py          ← PIPELINE 7: Upload model to GCS
│
├── models/
│   ├── emotion_model.py       ← infer_emotion(), detect_with_calibration()
│   └── face_model.py          ← FaceRecognizer (SVM + cosine gate wrapper)
│
├── storage/
│   └── gcs_storage.py         ← GCS upload/download abstraction
│
├── utils/
│   ├── logger.py              ← Structured logger (console + file + JSON)
│   └── image_utils.py         ← load/save/resize/crop helpers
│
├── api/
│   └── app.py                 ← FastAPI REST server
│
├── tests/
│   └── test_pipelines.py      ← Pytest unit tests
│
├── .vscode/
│   ├── settings.json          ← Python interpreter, lint, format, test config
│   └── launch.json            ← One debug config per pipeline / mode
│
├── main.py                    ← Master CLI orchestrator
├── requirements.txt
├── Dockerfile
├── .env.example               ← Copy → .env and fill in values
└── .gitignore
```

---

## Step-by-Step Setup in VS Code

### STEP 1 — Clone / Open the project

```bash
# Open VS Code, then open the face_emotion_system/ folder
# File → Open Folder → select face_emotion_system/
```

---

### STEP 2 — Create a virtual environment

Open the **integrated terminal** in VS Code (`Ctrl+`` ` ``):

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

VS Code will detect the new `.venv` and prompt you to select it as the
interpreter. Click **Yes**, or press `Ctrl+Shift+P` → "Python: Select Interpreter"
and pick `.venv/bin/python`.

---

### STEP 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Note on TensorFlow:** If you are on Apple Silicon (M1/M2/M3) use
> `tensorflow-macos` instead of `tensorflow` in requirements.txt.

---

### STEP 4 — Configure secrets (.env)

```bash
cp .env.example .env
```

Open `.env` and fill in:

```
GCS_PROJECT_ID=your-gcp-project-id
GCS_BUCKET_NAME=your-bucket-name
GCS_KEY_PATH=secrets/gcs_key.json
```

Then place your GCS service-account JSON file at `secrets/gcs_key.json`.

> ⚠️ **Security rule**: `secrets/` and `.env` are in `.gitignore`.
> Never commit credentials. The original notebook embedded a private key
> in source — this project fixes that anti-pattern completely.

---

### STEP 5 — Run the status check

```bash
python main.py status
```

You should see all model files listed as ❌ (not yet present). That is expected.

---

### STEP 6 — Run the full training pipeline

```bash
python main.py train
```

This runs all 7 pipeline stages in order:

| Stage | What happens |
|-------|-------------|
| **1. Data Ingestion** | Checks GCS for a saved model. If found, downloads it and exits early (no training). If not found, downloads TeamFaces.zip + raf-db.zip and extracts them. |
| **2. Preprocessing** | Walks `data/raw/TeamFaces/`, validates each image (HEIC conversion, resize, face sanity check), writes clean JPEGs to `data/processed/`. |
| **3. Feature Extraction** | Extracts 512-d FaceNet embeddings per image, augments with Gaussian noise, L2-normalises. Caches to `saved_models/embeddings_cache.pkl` so re-runs are instant. |
| **4. Training** | Stratified 80/20 split → CalibratedSVC(RBF, C=10) → 3-fold CV → evaluation report + confusion matrix. Saves 4 artefact files. |
| **5. Deployment** | Uploads all 5 model artefacts back to GCS bucket. Next `python main.py train` will load them instead of re-training. |

> To force re-training even if a saved model exists:
> ```bash
> python main.py train --force-retrain --skip-deploy
> ```

---

### STEP 7 — Run inference on a photo

```bash
python main.py infer --image path/to/your/group_photo.jpg
```

Output:
- Annotated image → `data/output/<name>_detected.jpg`
- JSON results    → `data/output/<name>_results.json`
- Console log showing each face: name, emotion, confidence

---

### STEP 8 — Real-time webcam (local only)

```bash
python main.py realtime
```

Controls in the OpenCV window:
- `q` — quit
- `s` — save snapshot to `data/output/snapshots/`

---

### STEP 9 — Start the REST API

```bash
python main.py api
# or directly:
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000/docs` for the auto-generated Swagger UI.

**Available endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/model/info` | Loaded model metadata |
| GET | `/metrics` | Request count, avg latency |
| POST | `/predict/image` | Upload a JPEG/PNG → get face results |
| POST | `/predict/base64` | Send base64 image string → get results |

**Example curl:**
```bash
curl -X POST http://localhost:8000/predict/image \
     -F "file=@path/to/photo.jpg"
```

---

### STEP 10 — Run tests

```bash
pytest tests/ -v
```

Or use the VS Code Testing panel (beaker icon in the left sidebar).

---

### STEP 11 — Docker (production)

```bash
# Build
docker build -t face-emotion-api .

# Run (pass your .env file)
docker run -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/secrets:/app/secrets:ro \
  -v $(pwd)/saved_models:/app/saved_models \
  face-emotion-api
```

---

## Debugging in VS Code

Open `.vscode/launch.json` — you have a **pre-configured debug config** for
every pipeline stage. To use them:

1. Open the **Run & Debug** panel (`Ctrl+Shift+D`)
2. Select a config from the dropdown at the top, e.g. `▶ Train (full pipeline)`
3. Press `F5`

You can set breakpoints in any pipeline file and the debugger will stop there.

---

## How Each Pipeline Maps to the Original Notebook

| Original Notebook Cell | Production File |
|------------------------|----------------|
| CELL 1 — pip install   | `requirements.txt` |
| CELL 2 — GCS auth + download | `pipelines/data_ingestion.py` + `storage/gcs_storage.py` |
| CELL 3 — imports       | Distributed to each module |
| CELL 4 — shared config + `infer_emotion()` | `config/settings.py` + `models/emotion_model.py` |
| CELL 5 — `detect_with_calibration()` | `models/emotion_model.py` |
| CELL 6 — train SVM     | `pipelines/preprocessing.py` + `pipelines/feature_extraction.py` + `pipelines/training.py` |
| CELL 6b — upload model | `pipelines/deployment.py` |
| CELL 7 — group photo   | `pipelines/inference.py` |
| CELL 8 — webcam loop   | `pipelines/realtime.py` |

---

## Key Security Fix

The original notebook contained a **hard-coded GCS private key** inside the
Python source. This is a critical security vulnerability — anyone with read
access to the file can impersonate the service account.

**This project fixes it by:**
1. Storing the key as a JSON file in `secrets/` (excluded from git).
2. Reading only the file PATH from an environment variable (`GCS_KEY_PATH`).
3. In production (Cloud Run / GKE), using Application Default Credentials
   instead of a key file — no secrets needed at all.

---

## Logging

Three log outputs are written simultaneously:

| File | Format | Purpose |
|------|--------|---------|
| console | Coloured human-readable | Development |
| `logs/app.log` | Plain-text rotating | Ops / debugging |
| `logs/app.jsonl` | One JSON object per line | Machine ingestion (ELK, Datadog, Cloud Logging) |

Log level is controlled by the `LOG_LEVEL` env var (`DEBUG`, `INFO`, `WARNING`, `ERROR`).
