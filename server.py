"""FastAPI server wrapping the BirdNET 2.4 bird sound classifier.

Exposes these endpoints:
  GET  /        — active configuration (labels, sample_rate, window_samples, model)
  GET  /health  — liveness check
  GET  /labels  — ordered list of BirdNET labels ("Scientific name_Common name") matching logit indices
  POST /logits  — send {"audio": "<base64 float32 PCM bytes>"}, receive classification logits

The model is the BirdNET GLOBAL 6K V2.4 FP32 TFLite file that ships inside the
`birdnetlib` package, with its labels file. No download is needed.

/logits returns the RAW model output, before BirdNET's sigmoid. birdnetlib's
confidences are `1 / (1 + exp(-clip(logit, -15, 15)))` (sensitivity 1.0); the
client applies the activation itself.

Deploy with:
  uvicorn server:app --host 0.0.0.0 --port 8003

Environment variables:
  BIRDNET_THREADS  number of TFLite interpreter threads (default: 1)
"""

import base64
import os
from contextlib import asynccontextmanager

import numpy as np
from birdnetlib.analyzer import LABEL_PATH, MODEL_PATH
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Shared state (populated at startup)
# ---------------------------------------------------------------------------
_state: dict = {}

SAMPLE_RATE = 48_000
WINDOW_SAMPLES = 3 * SAMPLE_RATE  # 144 000


def _load_interpreter(model_path: str, num_threads: int):
    """Load the TFLite model the way birdnetlib does (tflite_runtime if present, else TF)."""
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        from tensorflow import lite as tflite
    interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def _run_model(audio_batch: np.ndarray) -> np.ndarray:
    """Run the interpreter on [N, WINDOW_SAMPLES] clips and return raw logits [N, n_classes]."""
    interpreter = _state["interpreter"]
    input_index = _state["input_index"]
    if _state["batch_size"] != len(audio_batch):
        interpreter.resize_tensor_input(input_index, [len(audio_batch), WINDOW_SAMPLES])
        interpreter.allocate_tensors()
        _state["batch_size"] = len(audio_batch)
    interpreter.set_tensor(input_index, np.ascontiguousarray(audio_batch, dtype=np.float32))
    interpreter.invoke()
    return interpreter.get_tensor(_state["output_index"])  # [N, n_classes]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    num_threads = int(os.environ.get("BIRDNET_THREADS", "1"))
    interpreter = _load_interpreter(MODEL_PATH, num_threads)
    _state["interpreter"] = interpreter
    _state["input_index"] = interpreter.get_input_details()[0]["index"]
    _state["output_index"] = interpreter.get_output_details()[0]["index"]
    _state["batch_size"] = 1

    with open(LABEL_PATH) as f:
        _state["labels"] = [line.strip() for line in f if line.strip()]

    n_outputs = _run_model(np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)).shape[-1]
    if n_outputs != len(_state["labels"]):
        raise RuntimeError(f"Model has {n_outputs} outputs but the labels file has {len(_state['labels'])} lines.")

    print(f"BirdNET server ready. {len(_state['labels'])} classes, {num_threads} thread(s), model {os.path.basename(MODEL_PATH)}.")
    yield
    # --- shutdown: nothing to clean up ---


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def describe():
    """Return the active configuration of this adapter (used by HttpClient.describe())."""
    return {
        "labels": _state["labels"],
        "sample_rate": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "model": os.path.basename(MODEL_PATH),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels")
def labels():
    """Return the ordered list of BirdNET labels corresponding to logit indices."""
    return _state["labels"]


class LogitsRequest(BaseModel):
    audio: str  # base64-encoded float32 PCM bytes


def _decode_clips(audio: str) -> np.ndarray:
    """Decode base64 float32 PCM into [N, WINDOW_SAMPLES] clips."""
    try:
        raw = base64.b64decode(audio)
    except Exception:
        raise HTTPException(status_code=422, detail="audio field is not valid base64.")

    n_clips, remainder = divmod(len(raw), WINDOW_SAMPLES * 4)
    if n_clips == 0 or remainder != 0:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Decoded audio length {len(raw)} B is not a positive multiple of "
                f"{WINDOW_SAMPLES * 4} B (1 clip = {WINDOW_SAMPLES} float32 samples)."
            ),
        )
    return np.frombuffer(raw, dtype=np.float32).reshape(n_clips, WINDOW_SAMPLES)


@app.post("/logits")
def logits(body: LogitsRequest):
    """Accept a batch of mono audio clips as base64-encoded float32 PCM bytes.

    The decoded bytes must be N * WINDOW_SAMPLES * 4 bytes: N clips concatenated,
    each exactly WINDOW_SAMPLES float32 samples (3 s at 48 kHz, little-endian).
    Batch size N is inferred from the decoded payload length.
    Returns {"logits": [[float, ...]]} with shape [N, n_classes], before the sigmoid.
    """
    audio_batch = _decode_clips(body.audio)
    return {"logits": _run_model(audio_batch).tolist()}
