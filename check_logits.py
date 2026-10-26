"""Checks for the /logits endpoint; run on its own in a Slurm job.

Usage:
  .venv/bin/python check_logits.py AUDIO [AUDIO ...]

For each audio file:
  - birdnetlib reads it and cuts it into 3 s chunks (its own loader: 48 kHz mono,
    last chunk zero-padded);
  - birdnetlib's Analyzer.predict gives its confidence for each chunk;
  - the same chunks go through this server's /logits over HTTP (FastAPI TestClient);
  - sigmoid(clip(logit, -15, 15)) must equal birdnetlib's confidence.
Prints the max difference and the top 3 labels of the loudest chunk.
"""

import base64
import sys

import numpy as np
from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from fastapi.testclient import TestClient

import server

TOL = 1e-5


def sigmoid(x: np.ndarray) -> np.ndarray:
    """BirdNET's flat sigmoid at sensitivity 1.0."""
    return 1 / (1.0 + np.exp(-np.clip(x, -15, 15)))


def main() -> None:
    analyzer = Analyzer()
    worst = 0.0
    with TestClient(server.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        labels = client.get("/labels").json()
        assert labels == analyzer.labels, "server labels differ from birdnetlib's"

        for path in sys.argv[1:]:
            rec = Recording(analyzer, path, min_conf=0.0)
            rec.read_audio_data()
            chunks = np.stack([np.asarray(c, dtype=np.float32) for c in rec.chunks])
            assert chunks.shape[1] == server.WINDOW_SAMPLES, chunks.shape

            expected = np.concatenate([analyzer.predict(c) for c in chunks])  # [N, n_classes]
            body = {"audio": base64.b64encode(chunks.tobytes()).decode()}
            logits = np.asarray(client.post("/logits", json=body).json()["logits"])
            assert logits.shape == expected.shape, (logits.shape, expected.shape)

            diff = np.abs(sigmoid(logits) - expected).max()
            worst = max(worst, diff)
            loudest = int(np.argmax(np.abs(chunks).max(axis=1)))
            top = np.argsort(logits[loudest])[::-1][:3]
            tops = ", ".join(f"{labels[i]} ({expected[loudest, i]:.2f})" for i in top)
            print(f"{path}: {len(chunks)} chunks, max |sigmoid(logits) - birdnetlib| = {diff:.2e}; chunk {loudest}: {tops}")

        bad = client.post("/logits", json={"audio": base64.b64encode(b"\0" * 12).decode()})
        assert bad.status_code == 422, bad.status_code

    assert worst <= TOL, f"check FAILED: {worst:.2e} > {TOL}"
    print(f"check PASSED: max difference {worst:.2e} <= {TOL}")


if __name__ == "__main__":
    main()
