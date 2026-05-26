#!/bin/bash
# One-shot setup for Qwen3.6-35B-A3B MTP on GCP L4
# Run on the GCP instance after creation
set -e

echo "=== Installing Docker ==="
if ! command -v docker &>/dev/null; then
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
fi

echo "=== Downloading model (MTP version from unsloth) ==="
mkdir -p ~/models
if [ ! -f ~/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf ]; then
    pip install -q huggingface-hub
    # IMPORTANT: Must use the MTP-specific repo, NOT the regular GGUF repo
    # Regular repo: unsloth/Qwen3.6-35B-A3B-GGUF (no MTP layers)
    # MTP repo:     unsloth/Qwen3.6-35B-A3B-MTP-GGUF (with MTP layers)
    huggingface-cli download unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
        Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
        --local-dir ~/models \
        --local-dir-use-symlinks False
    # Rename to include MTP in filename for clarity
    mv ~/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf ~/models/Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf
else
    echo "Model already downloaded"
fi

echo "=== Pulling Docker image ==="
sudo docker pull ghcr.io/ggml-org/llama.cpp:server-cuda

echo "=== Starting services ==="
cd "$(dirname "$0")/.."
sudo docker compose up -d

echo "=== Waiting for llama-server to be healthy ==="
for i in $(seq 1 60); do
    if curl -s http://localhost:8080/health | grep -q ok; then
        echo "llama-server is ready!"
        break
    fi
    echo "Waiting... ($i/60)"
    sleep 5
done

IP=$(curl -s ifconfig.me)
echo ""
echo "=== Deployment complete ==="
echo "Web UI:  http://$IP:8080"
echo "API:     http://$IP:8080/v1/chat/completions"
echo ""
echo "Test:"
echo "  curl http://$IP:8080/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"max_tokens\":256}'"
echo ""
echo "Stop: gcloud compute instances stop qwen36-mtp-l4 --project=jinaai-dev --zone=\$ZONE"
