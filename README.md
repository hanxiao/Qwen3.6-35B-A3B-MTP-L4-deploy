# Qwen3.6-35B-A3B MTP on GCP L4 (24GB VRAM)

Deploy Qwen3.6-35B-A3B with **Multi-Token Prediction (MTP)** speculative decoding on a single NVIDIA L4 GPU using llama.cpp server.

## Key Results

| Metric | Value |
|--------|-------|
| Decode speed | ~70 tok/s |
| MTP acceptance rate | ~83% |
| Quantization | Q4_K_XL (Unsloth Dynamic 2.0) |
| Model size | ~22 GB |
| VRAM usage | ~21.3 / 22.6 GB (94%) |
| Context window | 8192 tokens |
| Thinking mode | Enabled |
| Built-in tools | Enabled (`--tools all`) |

## Quick Start

### 1. Create GCP Instance

```bash
gcloud compute instances create qwen36-mtp-l4 \
  --project=$GCP_PROJECT \
  --zone=us-west1-a \
  --machine-type=g2-standard-8 \
  --image=pytorch-2-7-cu128-ubuntu-2204-nvidia-570-v20260219 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE
```

- **Standard instance** (not spot) - MTP model loading takes time, preemption wastes it
- g2-standard-8: 1x NVIDIA L4 (24GB), 8 vCPUs, 32GB RAM

### 2. Deploy

```bash
git clone https://github.com/hanxiao/Qwen3.6-35B-A3B-MTP-L4-deploy.git
cd Qwen3.6-35B-A3B-MTP-L4-deploy
./scripts/setup.sh
```

### 3. Firewall

```bash
gcloud compute firewall-rules create allow-llama-8080 \
  --project=$GCP_PROJECT \
  --allow=tcp:8080 --target-tags=llama-server
```

## Architecture

```
llama.cpp server (:8080)  [OpenAI-compatible API + Built-in Web UI + Tools]
  └── MTP speculative decoding (draft-mtp, n_max=2)
  └── Jinja chat template (patched think block handling)
```

No Open WebUI needed - llama-server has a built-in chat UI at `:8080`.

## Lessons Learned / Gotchas

### 1. MTP Model Source (CRITICAL)

The MTP draft layers are baked into the model weights. You **must** download from the MTP-specific HuggingFace repo:

| Repo | MTP layers | Use for |
|------|-----------|---------|
| `unsloth/Qwen3.6-35B-A3B-GGUF` | ❌ No | Regular inference |
| `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` | ✅ Yes | MTP speculative decoding |

If you use the wrong repo, llama.cpp will error: `model doesn't contain MTP layers`.

### 2. Memory Fitting (`-fitt`)

Q4_K_XL MTP model (22GB) + MTP context (529 MiB) + KV cache is extremely tight on L4 (22.6GB VRAM).

- Default `--fit` target margin is **1024 MiB** - too conservative, pushes many layers to CPU → **13 tok/s**
- Set `-fitt 256` to reduce margin → all 42/42 layers on GPU → **70 tok/s**
- With `-fit off -ngl 99`: OOM (MTP compute buffer ~513 MiB can't fit)
- With `--parallel 4` (default auto): OOM or severe CPU offload

**Rule: always use `-fitt 256 --parallel 1`** for this model on L4.

### 3. llama.cpp Flag Changes

Flags change between llama.cpp versions. Current (2026-05 build):

| Old flag | New flag |
|----------|----------|
| `--draft-mtp` | `--spec-type draft-mtp` |
| `--ngl` | `-ngl` (short only) or `--gpu-layers` |
| `--n-parallel` | `--parallel` |
| `--spec-draft-n-max N` | same (unchanged) |

Always check `llama-server --help` when using a new image version.

### 4. `--no-mmap` for CPU-offloaded tensors

When auto-fit offloads some tensor overrides to CPU, llama.cpp warns:
> tensor overrides to CPU are used with mmap enabled - consider using --no-mmap

Always add `--no-mmap` to avoid performance degradation.

### 5. `--tools all` for Built-in Tools

Add `--tools all` to enable llama-server's built-in tool calling support. Without it, the Web UI won't show tool capabilities.

### 6. Thinking Mode

Thinking is enabled by default (`thinking = 1`). To disable, add to docker-compose command:
```
--chat-template-kwargs '{"enable_thinking": false}'
```

### 7. Cost

| Mode | $/hr | $/month |
|------|------|---------|
| Standard | ~$0.86 | ~$620 |
| Spot | ~$0.26 | ~$190 |

Standard recommended for daily-use serving (spot gets preempted).

**Always stop when not in use:**
```bash
gcloud compute instances stop qwen36-mtp-l4 --project=jinaai-dev --zone=us-west1-a
```

## File Structure

```
├── README.md
├── docker-compose.yml           # All llama-server params pinned here
├── templates/
│   └── chat_template.jinja      # Patched Jinja template (think block fix)
└── scripts/
    └── setup.sh                 # One-shot: install Docker, download model, start
```
