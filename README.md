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

## Built-in Tools

`--tools all` enables server-side tool execution. Available tools:

| Tool | Description | Write |
|------|-------------|-------|
| `read_file` | Read file contents | No |
| `file_glob_search` | Find files by glob pattern | No |
| `grep_search` | Search file contents | No |
| `exec_shell_command` | Execute shell commands | Yes |
| `write_file` | Write/create files | Yes |
| `edit_file` | Edit file contents | Yes |
| `apply_diff` | Apply unified diffs | Yes |
| `get_datetime` | Get current date/time | No |

### How Built-in Tools Work

The server exposes two endpoints:
- `GET /tools` — returns tool definitions (OpenAI function-calling format)
- `POST /tools` — executes a tool call on the server

**The server does NOT auto-execute tools.** The client must run the agentic loop:

```
Client                          Server
  │                               │
  │  POST /v1/chat/completions    │
  │  (messages + tools)           │
  │──────────────────────────────>│
  │                               │
  │  Response with tool_calls     │
  │<──────────────────────────────│
  │                               │
  │  POST /tools                  │
  │  (execute tool)               │
  │──────────────────────────────>│
  │                               │
  │  Tool result                  │
  │<──────────────────────────────│
  │                               │
  │  POST /v1/chat/completions    │
  │  (messages + tool result)     │
  │──────────────────────────────>│
  │                               │
  │  Final answer                 │
  │<──────────────────────────────│
```

The Web UI does this loop automatically. For API usage, use `scripts/chat_with_tools.py`.

### API Examples

#### Step 1: Fetch tool definitions

```bash
curl http://SERVER:8080/tools
```

Response:
```json
[
  {
    "display_name": "Get Date & Time",
    "tool": "get_datetime",
    "type": "builtin",
    "definition": {
      "type": "function",
      "function": {
        "name": "get_datetime",
        "description": "Returns the current date and time"
      }
    }
  },
  {
    "display_name": "Execute shell command",
    "tool": "exec_shell_command",
    "type": "builtin",
    "definition": {
      "type": "function",
      "function": {
        "name": "exec_shell_command",
        "description": "Execute a shell command and return its output",
        "parameters": {
          "type": "object",
          "properties": {
            "command": {
              "type": "string",
              "description": "The shell command to execute"
            }
          },
          "required": ["command"]
        }
      }
    }
  }
]
```

#### Step 2: Send chat request with tools

Extract the `definition` field from each tool and pass as `tools` array:

```bash
curl http://SERVER:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [
      {"role": "user", "content": "What is the current date and time?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_datetime",
          "description": "Returns the current date and time"
        }
      }
    ],
    "max_tokens": 4096
  }'
```

Response (model decides to call a tool):
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "reasoning_content": "The user is asking for the current date and time. I have a tool called get_datetime that can provide this.",
      "content": "",
      "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
          "name": "get_datetime",
          "arguments": "{}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

#### Step 3: Execute tool on server

```bash
curl -X POST http://SERVER:8080/tools \
  -H "Content-Type: application/json" \
  -d '{"tool": "get_datetime", "params": {}}'
```

Response:
```json
{
  "result": "Tue May 26 19:35:51 2026\n"
}
```

#### Step 4: Send tool result back

```bash
curl http://SERVER:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6",
    "messages": [
      {"role": "user", "content": "What is the current date and time?"},
      {"role": "assistant", "content": "", "tool_calls": [{"id": "call_abc123", "type": "function", "function": {"name": "get_datetime", "arguments": "{}"}}]},
      {"role": "tool", "tool_call_id": "call_abc123", "content": "{\"result\": \"Tue May 26 19:35:51 2026\\n\"}"}
    ],
    "tools": [
      {"type": "function", "function": {"name": "get_datetime", "description": "Returns the current date and time"}}
    ],
    "max_tokens": 4096
  }'
```

Response (final answer):
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "The current date and time is Tuesday, May 26, 2026, at 19:35:51."
    },
    "finish_reason": "stop"
  }]
}
```

### Python Client

`scripts/chat_with_tools.py` automates the full agentic loop:

```bash
python scripts/chat_with_tools.py "What time is it?"
python scripts/chat_with_tools.py "List files in /models and their sizes"
python scripts/chat_with_tools.py --base-url http://YOUR_IP:8080 "your question"
```

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

### 5. Built-in Tools Architecture

`--tools all` does NOT make the server auto-execute tools. It:
- Exposes `GET /tools` (list definitions) and `POST /tools` (execute)
- The Web UI runs the agentic loop client-side (fetch tools → inject into request → intercept tool_calls → POST /tools → loop)
- API clients must implement the same loop (see `scripts/chat_with_tools.py`)

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
├── docker-compose.yml             # All llama-server params pinned here
├── templates/
│   └── chat_template.jinja        # Patched Jinja template (think block fix)
└── scripts/
    ├── setup.sh                   # One-shot: install Docker, download model, start
    └── chat_with_tools.py         # API client with built-in tools agentic loop
```
