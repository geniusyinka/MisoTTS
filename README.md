<div align="center">

<img src="images/repo_banner.png" alt="Miso TTS 8B" width="100%">

# Miso TTS 8B

### State-of-the-Art Text-to-Speech Model

<p>
  <a href="https://misolabs.ai"><img alt="Website" src="https://img.shields.io/badge/Website-misolabs.ai-black?style=for-the-badge"></a>
  <a href="https://huggingface.co/MisoLabs/MisoTTS"><img alt="Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-MisoTTS-yellow?style=for-the-badge"></a>
  <a href="https://github.com/MisoLabsAI"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-MisoLabsAI-181717?style=for-the-badge&logo=github&labelColor=555555"></a>
  <a href="https://x.com/MisoLabsAI"><img alt="X" src="https://img.shields.io/badge/-MisoLabsAI-181717?style=for-the-badge&logo=x&labelColor=555555"></a>
</p>

<p>
  <a href="#quickstart">Quickstart</a> |
  <a href="#model-introduction">Model Introduction</a> |
  <a href="#model-summary">Model Summary</a> |
  <a href="#usage">Usage</a> |
  <a href="#apple-silicon-mps--mlx">Apple Silicon</a> |
  <a href="#safety">Safety</a>
</p>

</div>

---

## Quickstart

To quickly try the model, you can use the demo hosted on our [landing page](https://misolabs.ai)
at misolabs.ai. To try it locally, follow the instructions below.

If you do not have `uv` installed yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then clone the repository and create the environment:

```bash
git clone https://github.com/MisoLabsAI/MisoTTS.git
cd MisoTTS
uv sync --python 3.10
source .venv/bin/activate
```

Then run the example conversation. By default, `run_misotts.py` loads the public
model from [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS) and
downloads it into the Hugging Face cache if it is not already present on your
machine:

```bash
uv run python run_misotts.py
```

The script writes `full_conversation.wav` in the repository root.

With `pip` instead of `uv`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
python run_misotts.py
```

---

## Model Introduction

Miso TTS 8B is a text-to-dialogue RVQ Transformer inspired by the Sesame CSM architecture. It
generates Mimi audio codes from text and optional audio context, using a large
Llama 3.2-style backbone and a smaller autoregressive audio decoder. To find out more
about the architecture, read [our blog post](https://misolabs.ai/blog/miso-tts-8b).

The model is designed for high-quality conversational speech generation.
This repository contains the inference
code, model definition, and setup instructions for running Miso TTS locally.

---

## Model Summary

| Item                | Value           |
| ------------------- | --------------- |
| Model               | Miso TTS 8B     |
| Organization        | Miso Labs       |
| Task                | Text-to-speech  |
| Architecture        | RVQ Transformer |
| Backbone            | `llama-8B`      |
| Audio decoder       | `llama-300M`    |
| Text vocabulary     | `128,256`       |
| Audio vocabulary    | `2,051`         |
| Audio codebooks     | `32`            |
| Audio tokenizer     | Mimi            |
| Max sequence length | `2,048`         |

### Architecture

Miso TTS 8B uses two transformer components:

- A large backbone transformer that consumes text/audio-frame embeddings.
- A smaller decoder transformer that autoregressively predicts higher-order
  audio codebooks within each frame.

The backbone accepts interleaved text and audio tokens, allowing it to condition its generations on
the conversation history.

---

## Usage

### Python

```python
import torch
import torchaudio

from generator import load_miso_8b

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = load_miso_8b(
    device=device,
    model_path_or_repo_id="MisoLabs/MisoTTS",
)

audio = generator.generate(
    text="Hello from Miso.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("miso.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

### Prompted generation

Miso TTS can condition on prior audio for voice cloning.
This is optional; the quickstart example above runs without
prompt audio.

```python
import torchaudio

from generator import Segment, load_miso_8b

generator = load_miso_8b(device="cuda")

prompt_audio, sample_rate = torchaudio.load("prompt.wav")
prompt_audio = torchaudio.functional.resample(
    prompt_audio.squeeze(0),
    orig_freq=sample_rate,
    new_freq=generator.sample_rate,
)

context = [
    Segment(
        speaker=0,
        text="This is the transcript for the prompt audio.",
        audio=prompt_audio,
    )
]

audio = generator.generate(
    text="This is the next sentence to synthesize.",
    speaker=0,
    context=context,
    max_audio_length_ms=10_000,
)
```

---

## Weights

The model weights are hosted publicly on Hugging Face:

```bash
uv run python run_misotts.py
```

The default model repository is
[MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS). The first run
downloads the model automatically through Hugging Face Hub; later runs reuse the
cached copy.

The first run also downloads the SilentCipher watermarking model from
`sony/silentcipher`. If that separate download times out, rerun the command; the
Hugging Face cache resumes from files that already completed.

---

## Apple Silicon (MPS & MLX)

The upstream inference path is CUDA-only and explicitly skips Metal. This fork adds
**Apple Silicon support**, validated against the CUDA reference at the logit level
(not just "it makes sound") — teacher-forced frame-by-frame across **5 prompt styles and
1,117 frames** (incl. two 30s+ clips). Two paths are included:

- **PyTorch / MPS** (`run_misotts_mac.py`) — a correctness-first quick-win using
  `PYTORCH_ENABLE_MPS_FALLBACK=1`. Runs, but slow (~21× slower than real-time).
- **MLX** (`run_misotts_mlx.py`) — a native [MLX](https://github.com/ml-explore/mlx)
  port (model in `misotts_mlx.py`), ~**12.8× faster** than the MPS path:

  | Mode | Flag | RTF | Quality |
  |---|---|---|---|
  | fp32 | *(default)* | 4.71× | clean |
  | **Q8** | `--bits 8` | **2.01×** | **lossless** |
  | mixed-Q4 | `--quant mixed` | 1.65× | slight tradeoff |

  *(RTF = compute-seconds / audio-seconds; measured on an M4 Pro 64 GB. RTF > 1 means
  slower than real-time. fp32 & Q8 reproduce the CUDA logits to mean cosine 0.9999 / ~99%
  top-1 over 1,117 frames; Q8 is lossless even on 30s clips. mixed-Q4 trades fidelity
  (cos ~0.997, ~93% top-1) for speed.)*

```bash
uv pip install -r requirements-mlx.txt
python run_misotts_mlx.py --text "Hey! This is running on my Mac." --bits 8
```

The MLX path needs **no Hugging Face token** (it uses an ungated Llama-3.2 tokenizer
mirror and the public Mimi codec). See
[`docs/running-misotts-on-apple-silicon.md`](docs/running-misotts-on-apple-silicon.md)
for the full guide and gotchas, and
[`docs/porting-misotts-to-apple-silicon-technical.md`](docs/porting-misotts-to-apple-silicon-technical.md)
for the technical account (validation methodology, the port, and the real-time wall).

> **Note:** the MLX runner applies the SilentCipher watermark by default (CPU/torch, ~0.7s,
> imperceptible — verifiable with `python watermarking.py --audio_path <file>`); pass
> `--no-watermark` only for local debugging (see [Safety](#safety)). Generation is not
> real-time; see the table.

## Deployment Notes

Miso TTS 8B is a large model. For best results, use a CUDA GPU with sufficient
VRAM for the checkpoint precision you are loading. The default inference path
uses `torch.bfloat16`. On Apple Silicon, see [Apple Silicon](#apple-silicon-mps--mlx).

---

## Safety

Miso TTS is a speech generation model. Do not use it to impersonate people,
create deceptive audio, commit fraud, or generate harmful content.

Generated audio is watermarked by default. If you deploy this model in another
application, use your own private watermark key and keep it secret.

---

## Links

- Website: [misolabs.ai](https://misolabs.ai)
- Hugging Face: [MisoLabs/MisoTTS](https://huggingface.co/MisoLabs/MisoTTS)
- GitHub: [MisoLabsAI](https://github.com/MisoLabsAI)
- X: [@MisoLabsAI](https://x.com/MisoLabsAI)
