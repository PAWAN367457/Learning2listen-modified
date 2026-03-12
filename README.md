# Learning2Listen — Custom Dataset Training
# This repository is based on the original implementation of:Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion  
https://github.com/evonneng/learning2listen
# Modifications were made for experimentation and research purposes

Train the [Learning2Listen (L2L)](https://github.com/evonneng/learning2listen) model on **your own speaker/listener video dataset**.  
L2L generates realistic listener facial reactions given a speaker's audio and face motion.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Requirements](#requirements)
- [Repository Setup](#repository-setup)
- [Preparing Your Dataset](#preparing-your-dataset)
  - [Required File Format](#required-file-format)
  - [Extracting Face Blendshapes](#extracting-face-blendshapes)
  - [Extracting Audio Features](#extracting-audio-features)
  - [Directory Structure](#directory-structure)
- [Configuration Files](#configuration-files)
- [Training](#training)
  - [Stage 1 — VQ-VAE](#stage-1--vq-vae)
  - [Stage 2 — Listener Predictor](#stage-2--listener-predictor)
  - [Running the Script](#running-the-script)
- [Checkpoints](#checkpoints)
- [Using Real Listener Data](#using-real-listener-data)
- [Troubleshooting](#troubleshooting)
- [Key Constants Reference](#key-constants-reference)

---

## Overview

This repo provides a self-contained training script (`train_l2l_custom.py`) that lets you train L2L on your own conversational video data instead of the original Conan O'Brien dataset.

The training has two sequential stages:

1. **VQ-VAE** — learns a discrete codebook (vocabulary) of facial expressions from speaker face data.
2. **FACTModel Predictor** — learns to predict future listener reactions from the quantized speaker tokens + audio.

---

## Architecture

```
Speaker Audio (.npy)  ──┐
                        ├──► VQ-VAE Encoder ──► Codebook Tokens ──► FACTModel ──► Listener Face
Speaker Faces (.npy)  ──┘         (Stage 1)                          (Stage 2)
                                                        ▲
                                               Listener Past Tokens
```

**VQ-VAE** (`VQModelTransformer`):
- `TransformerEncoder` — Conv1d downsampler + Transformer → latent `z`
- `VectorQuantizer` — snaps `z` to nearest codebook entry
- `TransformerDecoder` — upsamples quantized tokens back to face sequence

**FACTModel Predictor**:
- Embeds listener past tokens
- Cross-modal Transformer fuses speaker audio + motion
- Predicts next listener face tokens autoregressively

---

## Requirements

```bash
# Python 3.9+ recommended
conda create -n l2l python=3.9
conda activate l2l

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy tqdm pyyaml librosa opencv-python
```

Clone the original L2L repo (model code lives there):

```bash
git clone https://github.com/evonneng/learning2listen
cd learning2listen
```

---

## Repository Setup

```
learning2listen/
├── src/
│   ├── train_l2l_custom.py        ← this training script
│   ├── vqgan/
│   │   ├── configs/
│   │   │   └── l2_32_smoothSS.json   ← VQ-VAE config
│   │   └── vqmodules/
│   │       ├── gan_models.py
│   │       └── quantizer.py
│   ├── modules/
│   │   ├── fact_model.py
│   │   └── base_models.py
│   └── configs/
│       └── vq/
│           └── delta_v6.json         ← Predictor config
├── checkpoints_custom/               ← saved during training
└── models/                           ← pretrained stats (optional)
```

Place `train_l2l_custom.py` inside `learning2listen/src/` and run it from there.

---

## Preparing Your Dataset

### Required File Format

For each video clip in your dataset you need **two `.npy` files** saved in the same folder:

| File | Shape | Description |
|---|---|---|
| `<clip_id>_speak_faces.npy` | `(T, 184)` | Speaker face blendshapes, T frames |
| `<clip_id>_speak_audio.npy` | `(4T, 128)` | Mel-spectrogram features, 4× denser than video |

- `T` = number of video frames (any length ≥ 32 works; longer clips are randomly cropped)
- `184` = full ARKit/MediaPipe blendshape vector (only first 56 channels are used)
- `128` = mel-filterbank bins

---

### Extracting Face Blendshapes

Use any face capture tool that outputs per-frame blendshape coefficients.  
**MediaPipe FaceMesh** example (outputs ~468 landmarks; adapt to your blendshape tool):

```python
import cv2, numpy as np, mediapipe as mp

def extract_face_features(video_path, out_path):
    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=False)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_face.process(rgb)
        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            vec = np.array([[l.x, l.y, l.z] for l in lm]).flatten()[:184]
        else:
            vec = np.zeros(184)
        frames.append(vec)
    cap.release()
    np.save(out_path, np.array(frames))   # shape: (T, 184)

extract_face_features("clip01_speaker.mp4", "clip01_speak_faces.npy")
```

> **Note:** The model only uses the first **56 channels** (`CHANNEL_SLICE=56`).  
> These correspond to expression blendshapes (0:50), head rotation (50:53), and jaw (53:56).  
> Make sure your feature vector puts the most expressive dimensions first.

---

### Extracting Audio Features

The model expects 128-dim mel-spectrogram features at **4× the video frame rate**:

```python
import librosa, numpy as np

def extract_audio_features(video_path, out_path, video_fps=25):
    """
    Extracts mel-spectrogram at 4x video frame rate.
    audio_fps = video_fps * 4 = 100 feature frames per second for 25fps video.
    """
    y, sr = librosa.load(video_path, sr=16000)
    hop_length = sr // (video_fps * 4)   # 4 audio frames per video frame
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max).T  # (4T, 128)
    np.save(out_path, mel_db.astype(np.float32))

extract_audio_features("clip01_speaker.mp4", "clip01_speak_audio.npy")
```

---

### Directory Structure

All `.npy` files go in a single flat folder (set as `DATA_ROOT` in the script):

```
l2l_inputs/
├── clip001_speak_faces.npy
├── clip001_speak_audio.npy
├── clip002_speak_faces.npy
├── clip002_speak_audio.npy
├── clip003_speak_faces.npy
├── clip003_speak_audio.npy
└── ...
```

The script auto-discovers all `*_speak_faces.npy` files and matches them to their audio pair.  
A 90/10 train/val split is applied automatically.

---

## Configuration Files

Two JSON configs drive the model architecture. They live inside the L2L repo and you generally **do not need to edit them**.

### `vqgan/configs/l2_32_smoothSS.json` (VQ-VAE)

Key values the training script reads:

| Field | Value | Meaning |
|---|---|---|
| `transformer_config.in_dim` | `56` | Face channels consumed by encoder |
| `transformer_config.quant_sequence_length` | `4` | Length after full downsampling |
| `transformer_config.quant_factor` | `3` | Downsampling exponent: SEQ_LEN = 4×2³ = **32** |
| `transformer_config.hidden_size` | `256` | Transformer width |
| `VQuantizer.n_embed` | `200` | Codebook size (number of discrete tokens) |
| `VQuantizer.zquant_dim` | `256` | Codebook embedding dimension |

### `configs/vq/delta_v6.json` (Predictor)

| Field | Value | Meaning |
|---|---|---|
| `fact_model.cross_modal_model.max_mask_len` | `4` | Masking window during training |
| `fact_model.speaker_full_transformer_config.sequence_length` | `40` | Speaker motion frames fed to predictor |
| `fact_model.listener_past_transformer_config.sequence_length` | `4` | Listener past context length |

> `AUDIO_SEQ_LEN = speaker_seq_len × 4 = 160` because the model applies `MaxPool1d(4)` on audio internally.

---

## Training

### Stage 1 — VQ-VAE

Trains the face codebook on **speaker** face data only.

- Input: `(B, 32, 56)` — 32 frames, 56 channels
- Output: reconstructed faces + quantization loss
- Loss = `L1(expression) + L1(rotation) + L1(jaw) + quantization_loss`
- Saves checkpoints every 10 epochs + best model

### Stage 2 — Listener Predictor

Runs after Stage 1 completes. The VQ-VAE is **frozen**.

- Input dict to `FACTModel.forward()`:

  | Key | Shape | Description |
  |---|---|---|
  | `listener_past` | `(B, 4)` long | Past listener token indices |
  | `speaker_full` | `(B, 40, 56)` float | Speaker face motion |
  | `audio_full` | `(B, 160, 128)` float | Raw audio features |

- Loss: cross-entropy over predicted listener codebook tokens
- Saves checkpoints every 5 epochs + best model

> **Currently**: `listener_past` and `listener_future` are **zero-filled placeholders**.  
> See [Using Real Listener Data](#using-real-listener-data) to make Stage 2 meaningful.

---

### Running the Script

**Step 1** — Edit the path constants at the top of `train_l2l_custom.py`:

```python
REPO_ROOT        = Path("/path/to/your/learning2listen")
DATA_ROOT        = Path("/path/to/your/l2l_inputs")          # folder with .npy files
PRED_CONFIG_PATH = Path(".../src/configs/vq/delta_v6.json")
```

**Step 2** — Adjust hyperparameters if needed:

```python
BATCH_SIZE  = 32     # reduce to 8-16 if GPU OOM
EPOCHS_VQ   = 100
EPOCHS_PRED = 50
LR_VQ       = 2e-4
LR_PRED     = 1e-4
```

**Step 3** — Run from the `src/` directory:

```bash
cd /path/to/learning2listen/src
python train_l2l_custom.py
```

Expected output:

```
Using device : cuda
SEQ_LEN      : 32
Config check passed — SEQ_LEN=32
Dataset: 120 clips in .../l2l_inputs
Train: 108 | Val: 12

========== Stage 1: VQ-VAE Training ==========
VQ  [  1/100]  total=7.213400  recon=0.312000
...
VQ  [100/100]  total=4.821480  recon=0.141057
VQ-VAE done.  Best loss: 4.800096

========== Stage 2: Predictor Training ==========
Pred [  1/50]  loss=5.298100
...
```

---

## Checkpoints

All checkpoints are saved to `learning2listen/checkpoints_custom/`:

```
checkpoints_custom/
├── vq_best.pt          ← VQ-VAE at lowest training loss
├── vq_final.pt         ← VQ-VAE after all epochs
├── vq_epoch_10.pt      ← VQ-VAE periodic saves
├── vq_epoch_20.pt
├── ...
├── pred_best.pt        ← Predictor at lowest training loss
├── pred_final.pt       ← Predictor after all epochs
├── pred_epoch_5.pt     ← Predictor periodic saves
└── ...
```

To **resume training** from a checkpoint, modify `setup_model` / `VQModelTransformer` calls to pass `load_path=`.

---

## Using Real Listener Data

Currently the script trains with **zero listener tokens** — a placeholder. To train a meaningful predictor you need real paired listener face data.

**Step 1** — Extract listener face blendshapes the same way as speaker:

```python
extract_face_features("clip01_listener.mp4", "clip01_listen_faces.npy")
```

**Step 2** — Quantize listener faces using the trained VQ-VAE to get token indices:

```python
import torch, numpy as np
from vqgan.vqmodules.gan_models import VQModelTransformer

vq_model = VQModelTransformer(l_vqconfig, version="")
vq_model.load_state_dict(torch.load("checkpoints_custom/vq_final.pt"))
vq_model.eval()

listen_faces = torch.from_numpy(np.load("clip01_listen_faces.npy")).float()
listen_faces = listen_faces[:32, :56].unsqueeze(0)   # (1, 32, 56)

with torch.no_grad():
    _, _, info = vq_model.encode(listen_faces)
    indices = info[2]   # (quant_seq_len,) — token indices for this clip
```

**Step 3** — Save as `<clip_id>_listen_tokens.npy` and load in the dataset's `__getitem__`:

```python
# In L2LDataset.__getitem__, replace the zero placeholders:
token_path = fp.with_name(fp.stem.replace("_speak_faces", "_listen_tokens") + fp.suffix)
listener_past    = np.load(token_path)[:LISTENER_PAST_LEN]      # (4,) int64
listener_future  = np.load(token_path)[LISTENER_PAST_LEN:LISTENER_PAST_LEN*2]  # (4,) int64
```

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `RuntimeError: size mismatch at non-singleton dimension 1` | SEQ_LEN doesn't match `quant_sequence_length × 2^quant_factor` | Let the script auto-derive SEQ_LEN from config (already done) |
| `AttributeError: no attribute 'training_step'` | Wrong method called on VQModelTransformer | Use `vq_model(faces)` which calls `forward()` |
| `TypeError: forward() got unexpected keyword argument 'max_mask_len'` | Wrong FACTModel call signature | Use `generator(inputs_dict, max_mask=4, mask_index=-1)` |
| `ImportError: circular import in fact_model.py` | `fact_model.py` imports itself | Remove the self-import line inside `fact_model.py` |
| `FileNotFoundError: No *_speak_faces.npy found` | Wrong DATA_ROOT path | Check DATA_ROOT points to the folder containing `.npy` files |
| `CUDA OOM` | Batch too large | Reduce `BATCH_SIZE` to 8 or 16 |
| `VQ total loss not decreasing` | Learning rate too high | Try `LR_VQ = 5e-5` |

---

## Key Constants Reference

| Constant | Value | Source |
|---|---|---|
| `SEQ_LEN` | `32` | `quant_seq_len(4) × 2^quant_factor(3)` |
| `CHANNEL_SLICE` | `56` | `transformer_config.in_dim` |
| `SPEAKER_SEQ_LEN` | `40` | `speaker_full_transformer_config.sequence_length` |
| `AUDIO_SEQ_LEN` | `160` | `SPEAKER_SEQ_LEN × 4` (MaxPool1d(4) inside model) |
| `LISTENER_PAST_LEN` | `4` | `quant_sequence_length` = quantized seq length |
| `max_mask_len` | `4` | `cross_modal_model.max_mask_len` |
| Codebook size | `200` | `VQuantizer.n_embed` |

---

## Citation

If you use this training code, please cite the original Learning2Listen paper:

```bibtex
@inproceedings{ng2022learning2listen,
  title     = {Learning to Listen: Modeling Non-Deterministic Dyadic Facial Motion},
  author    = {Ng, Evonne and Ginosar, Shiry and Darrell, Trevor and Joo, Hanbyul},
  booktitle = {CVPR},
  year      = {2022}
}
```
