#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 --github-token <token> --wandb-token <token> --accelerator <gpu|cpu|tpu> --num-dataset-workers <n>"
    echo ""
    echo "  --github-token        GitHub personal access token for cloning KMRD"
    echo "  --wandb-token         Weights & Biases API token"
    echo "  --accelerator         Hardware accelerator: gpu, cpu, or tpu"
    echo "  --num-dataset-workers Number of workers for build_dataset_cache.py"
    exit 1
}

TOKEN=""
WANDB_TOKEN=""
ACCELERATOR=""
NUM_WORKERS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --github-token)
            TOKEN="$2"; shift 2 ;;
        --wandb-token)
            WANDB_TOKEN="$2"; shift 2 ;;
        --accelerator)
            ACCELERATOR="$2"; shift 2 ;;
        --num-dataset-workers)
            NUM_WORKERS="$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            usage ;;
    esac
done

[[ -z "$TOKEN" ]]        && { echo "Error: --github-token is required"; usage; }
[[ -z "$WANDB_TOKEN" ]]  && { echo "Error: --wandb-token is required"; usage; }
[[ -z "$ACCELERATOR" ]]  && { echo "Error: --accelerator is required"; usage; }
[[ -z "$NUM_WORKERS" ]]  && { echo "Error: --num-dataset-workers is required"; usage; }

if [[ "$ACCELERATOR" != "gpu" && "$ACCELERATOR" != "cpu" && "$ACCELERATOR" != "tpu" ]]; then
    echo "Error: --accelerator must be one of: gpu, cpu, tpu"
    usage
fi

# 1. Clone KMRD
echo "==> Cloning KMRD..."
git clone "https://${TOKEN}@github.com/samirbraga/KMRD.git"
cd KMRD

# 2. Install uv
echo "==> Installing uv..."
sudo snap install astral-uv --classic

# 3. Install project dependencies
echo "==> Running uv sync --extra ${ACCELERATOR}..."
uv sync --extra "$ACCELERATOR"

# 4. Configure Weights & Biases
echo "==> Configuring W&B..."
uv run wandb login "$WANDB_TOKEN"

# 5. Download CATH data
echo "==> Downloading CATH data..."
cd data
bash download_cath.sh
cd ..

# 6. Build dataset cache
echo "==> Building dataset cache..."
uv run build_dataset_cache.py --num-workers "$NUM_WORKERS"

echo "==> Setup complete."
