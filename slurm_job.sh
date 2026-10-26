#!/bin/bash
#SBATCH --job-name=birdnet-server
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/%u/logs/birdnet-server-%j.out

# ---------------------------------------------------------------------------
# BirdNET 2.4 FastAPI server — persistent Slurm job
#
# Usage:
#   sbatch slurm_job.sh
#
# Clients discover this server by reading the address file written below.
# BirdNET 2.4 is a TFLite model; it runs on CPU, so no GPU is requested.
# ---------------------------------------------------------------------------

set -e

PORT=${PORT:-8003}
REPO_DIR="$HOME/birdnet-server"
ADDR_FILE="${BIRDNET_ADDR_FILE:-$REPO_DIR/server.addr}"

# Write hostname:port so clients can find us
mkdir -p "$(dirname "$ADDR_FILE")"
echo "$(hostname):$PORT" > "$ADDR_FILE"
echo "Server address: $(cat $ADDR_FILE)"

export BIRDNET_THREADS="${BIRDNET_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"

cd "$REPO_DIR"
mkdir -p logs

export UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
export UV_CACHE_DIR="$HOME/.cache/uv"
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python fastapi "uvicorn[standard]" soundfile librosa audioread \
  "birdnetlib==0.18.1" "tensorflow==2.21.0"

.venv/bin/uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
