# BirdNET 2.4 inference server

A small [FastAPI](https://fastapi.tiangolo.com/) service that serves the
[BirdNET](https://github.com/kahst/BirdNET-Analyzer) GLOBAL 6K V2.4 classifier
over HTTP, for use as an inference backend by
[esp-research](https://github.com/earthspecies/esp-research)'s sound event
detection evaluation (`type: birdnet`).

The model is the FP32 TFLite file that ships inside the
[`birdnetlib`](https://github.com/joeweiss/birdnetlib) package (pinned to
0.18.1), with its labels file. No download or checkpoint is needed.

## Endpoints

* `GET /` — active configuration (`labels`, `sample_rate`, `window_samples`, `model`)
* `GET /health` — liveness check
* `GET /labels` — ordered list of BirdNET labels (`"Scientific name_Common name"`) matching logit indices
* `POST /logits` — send `{"audio": "<base64 float32 PCM bytes>"}`, receive `{"logits": [[...]]}` of shape `[N, 6522]`

The audio is N clips of exactly 3 s at 48 kHz (144 000 float32 samples each,
little-endian), concatenated. The batch size N is inferred from the payload
length.

`/logits` returns the model's output **before** BirdNET's sigmoid. birdnetlib's
confidence is `1 / (1 + exp(-clip(logit, -15, 15)))` (sensitivity 1.0);
the client applies the activation itself.

## Running

On the Slurm cluster (CPU partition; TFLite needs no GPU):

```bash
sbatch slurm_job.sh
```

The job builds `.venv` with `uv`, writes `hostname:port` to `server.addr`
(override with `BIRDNET_ADDR_FILE`), and starts the server on port 8003
(override with `PORT`). `BIRDNET_THREADS` sets the interpreter threads (default:
the job's CPU count).

Locally:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" soundfile librosa audioread \
  "birdnetlib==0.18.1" "tensorflow==2.21.0"
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8003
```

## Self-check

`check_logits.py` checks that `sigmoid(/logits)` equals birdnetlib's own
confidences on the same 3 s chunks of real audio, and that the labels match:

```bash
uv pip install --python .venv/bin/python httpx resampy  # resampy: birdnetlib's own loader
.venv/bin/python check_logits.py some_recording.wav [more.wav ...]
```

## License

The BirdNET model and labels belong to the BirdNET team (see
[BirdNET-Analyzer](https://github.com/kahst/BirdNET-Analyzer)) and are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); they are
installed from `birdnetlib`, not distributed in this repository.
