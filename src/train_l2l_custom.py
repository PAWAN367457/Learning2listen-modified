# train_l2l_custom.py
# Run from: cd .../learning2listen/src && python train_l2l_custom.py

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
import yaml

# ======================================
#  PATHS
# ======================================
REPO_ROOT        = Path("/home/mudasir/Pawan/MPII/simple_tranformer/learning2listen")
DATA_ROOT        = Path("/home/mudasir/Pawan/MPII/simple_tranformer/STGNN/l2l_inputs")
PRED_CONFIG_PATH = Path("/home/mudasir/Pawan/MPII/simple_tranformer/learning2listen/src/configs/vq/delta_v6.json")
CHECKPOINT_DIR   = REPO_ROOT / "checkpoints_custom"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ======================================
#  CONSTANTS  (verified from config files)
# ======================================
# VQ config: quant_sequence_length=4, quant_factor=3  =>  SEQ_LEN = 4 * 2^3 = 32
SEQ_LEN       = 32    # frames per clip fed to VQ encoder
CHANNEL_SLICE = 56    # in_dim in VQ transformer_config

# FACTModel (predictor) sequence length expectations from delta_v6.json:
#   speaker_full_transformer_config.sequence_length = 40  (speaker motion frames)
#   audio: MaxPool1d(4) applied inside model, so raw audio input = 40 * 4 = 160
#   listener_past_transformer_config.sequence_length = 4  (listener past tokens)
SPEAKER_SEQ_LEN  = 40    # speaker motion frames fed to predictor
AUDIO_SEQ_LEN    = 160   # raw audio frames (MaxPool1d(4) -> 40 inside model)
LISTENER_PAST_LEN = 4    # listener past token length (quant_sequence_length)

BATCH_SIZE  = 32
EPOCHS_VQ   = 100
EPOCHS_PRED = 50
LR_VQ       = 2e-4
LR_PRED     = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device : {DEVICE}")

# ======================================
#  REPO SRC ON PATH
# ======================================
sys.path.insert(0, str(REPO_ROOT / "src"))

from vqgan.vqmodules.gan_models import VQModelTransformer
from modules.fact_model import setup_model, calc_logit_loss

# ======================================
#  LOAD CONFIGS
# ======================================
print("\nLoading configs...")

with open(PRED_CONFIG_PATH) as f:
    pred_cfg = yaml.safe_load(f)

l_vqconfig_path = REPO_ROOT / "src" / pred_cfg['l_vqconfig']
with open(l_vqconfig_path) as f:
    l_vqconfig = yaml.safe_load(f)

# Sanity-check SEQ_LEN matches config
cfg_seq = (l_vqconfig['transformer_config']['quant_sequence_length'] *
           (2 ** l_vqconfig['transformer_config']['quant_factor']))
assert cfg_seq == SEQ_LEN, \
    f"SEQ_LEN mismatch: hardcoded={SEQ_LEN} vs config-derived={cfg_seq}"

max_mask_len = pred_cfg['fact_model']['cross_modal_model']['max_mask_len']  # = 4

print(f"  SEQ_LEN (VQ input)        : {SEQ_LEN}")
print(f"  SPEAKER_SEQ_LEN (predictor): {SPEAKER_SEQ_LEN}")
print(f"  AUDIO_SEQ_LEN  (predictor) : {AUDIO_SEQ_LEN}")
print(f"  LISTENER_PAST_LEN          : {LISTENER_PAST_LEN}")
print(f"  max_mask_len               : {max_mask_len}")

# ======================================
#  DATASET
# ======================================
class L2LDataset(Dataset):
    """
    Loads speaker face + audio .npy clips and prepares inputs for both
    the VQ-VAE (Stage 1) and the FACTModel predictor (Stage 2).

    File naming (DATA_ROOT):
        <id>_speak_faces.npy   shape (T, 184)   — speaker face blendshapes
        <id>_speak_audio.npy   shape (4T, 128)  — mel-spectrogram features

    FACTModel.forward() expects inputs dict:
        "listener_past"  : (B, LISTENER_PAST_LEN)        long  — token indices
        "speaker_full"   : (B, SPEAKER_SEQ_LEN, 56)      float — speaker motion
        "audio_full"     : (B, AUDIO_SEQ_LEN, 128)       float — raw audio

    listener_past is zero-filled tokens (placeholder).
    Replace with real quantized listener indices when available.
    """
    def __init__(self, data_dir: Path, seq_len: int, channel_slice: int,
                 speaker_seq_len: int, audio_seq_len: int,
                 listener_past_len: int):
        self.seq_len          = seq_len
        self.channel_slice    = channel_slice
        self.speaker_seq_len  = speaker_seq_len
        self.audio_seq_len    = audio_seq_len
        self.listener_past_len = listener_past_len

        self.face_files = sorted(Path(data_dir).glob("*_speak_faces.npy"))
        if not self.face_files:
            raise FileNotFoundError(f"No *_speak_faces.npy found in {data_dir}")
        print(f"  Dataset: {len(self.face_files)} clips in {data_dir}")

    def __len__(self):
        return len(self.face_files)

    def _align(self, arr, target_len, axis=0):
        """ Truncate or edge-pad arr along axis to target_len. """
        cur = arr.shape[axis]
        if cur >= target_len:
            return arr[:target_len] if axis == 0 else arr[:, :target_len]
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, target_len - cur)
        return np.pad(arr, pad_width, mode='edge')

    def __getitem__(self, idx):
        fp = self.face_files[idx]
        ap = fp.with_name(fp.stem.replace("_speak_faces", "_speak_audio") + fp.suffix)

        faces = np.load(fp)   # (T, 184)
        audio = np.load(ap)   # (4T, 128)

        if faces.ndim == 3:   # drop batch dim if present
            faces = faces[0]
            audio = audio[0]

        T = faces.shape[0]

        # ---- Random crop for VQ input (SEQ_LEN=32) ----
        if T >= self.seq_len:
            start = np.random.randint(0, T - self.seq_len + 1)
            vq_faces = faces[start : start + self.seq_len, :self.channel_slice]
        else:
            vq_faces = self._align(faces, self.seq_len)[:, :self.channel_slice]

        # ---- Speaker motion for predictor (SPEAKER_SEQ_LEN=40) ----
        speaker_faces = self._align(faces, self.speaker_seq_len)[:, :self.channel_slice]

        # ---- Audio for predictor (AUDIO_SEQ_LEN=160, MaxPool1d(4)->40) ----
        speaker_audio = self._align(audio, self.audio_seq_len)   # (160, 128)

        # ---- Listener past token indices (LISTENER_PAST_LEN=4) ----
        # Zeros = "no listener context" — replace with real VQ indices when available
        listener_past = np.zeros(self.listener_past_len, dtype=np.int64)

        # ---- Listener future target for predictor loss ----
        # Using zero indices (n_embed=200 classes). Replace with real data.
        listener_future_tokens = np.zeros(self.listener_past_len, dtype=np.int64)

        return {
            # VQ-VAE inputs
            "vq_faces"       : torch.from_numpy(vq_faces).float(),           # (32, 56)

            # Predictor inputs
            "speaker_full"   : torch.from_numpy(speaker_faces).float(),      # (40, 56)
            "audio_full"     : torch.from_numpy(speaker_audio).float(),      # (160, 128)
            "listener_past"  : torch.from_numpy(listener_past).long(),       # (4,)

            # Predictor target
            "listener_future": torch.from_numpy(listener_future_tokens).long(), # (4,)
        }


print("\nBuilding datasets...")
full_ds   = L2LDataset(DATA_ROOT,
                       seq_len=SEQ_LEN,
                       channel_slice=CHANNEL_SLICE,
                       speaker_seq_len=SPEAKER_SEQ_LEN,
                       audio_seq_len=AUDIO_SEQ_LEN,
                       listener_past_len=LISTENER_PAST_LEN)
split_idx = int(len(full_ds) * 0.9)
train_ds  = torch.utils.data.Subset(full_ds, range(split_idx))
val_ds    = torch.utils.data.Subset(full_ds, range(split_idx, len(full_ds)))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=4, pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

# ======================================
#  BUILD MODELS
# ======================================
print("\nBuilding models...")

vq_model = VQModelTransformer(l_vqconfig, version="").to(DEVICE)

# setup_model wraps generator in DataParallel internally
generator, g_optimizer, _ = setup_model(
    pred_cfg, l_vqconfig, s_vqconfig=None, load_path=None)
generator = generator.to(DEVICE)

print("Models ready.")

# ======================================
#  VQ LOSS  (mirrors calc_vq_loss in gan_models.py)
# ======================================
def vq_loss_fn(dec, target, emb_loss, quant_w=1.0, jaw_alpha=1.0):
    """
    Splits reconstruction loss by facial channel region:
      0:50   expression blendshapes
      50:53  head rotation
      53:    jaw
    """
    recon = nn.L1Loss()(dec[:, :, :50], target[:, :, :50])
    if CHANNEL_SLICE > 50:
        recon += nn.L1Loss()(dec[:, :, 50:53], target[:, :, 50:53])
    if CHANNEL_SLICE > 53:
        recon += jaw_alpha * nn.L1Loss()(dec[:, :, 53:], target[:, :, 53:])
    return emb_loss.mean() * quant_w + recon, recon


# ======================================
#  STAGE 1 — VQ-VAE TRAINING
# ======================================
print("\n========== Stage 1: VQ-VAE Training ==========")
optimizer_vq = optim.AdamW(vq_model.parameters(), lr=LR_VQ)
best_vq_loss = float('inf')

for epoch in range(EPOCHS_VQ):
    vq_model.train()
    total_loss = total_recon = 0.0

    for batch in tqdm(train_loader, desc=f"VQ {epoch+1}/{EPOCHS_VQ}", leave=False):
        faces = batch["vq_faces"].to(DEVICE)    # (B, 32, 56)

        dec, emb_loss = vq_model(faces)         # dec: (B, 32, 56)
        loss, recon   = vq_loss_fn(dec, faces, emb_loss)

        optimizer_vq.zero_grad()
        loss.backward()
        optimizer_vq.step()

        total_loss  += loss.item()
        total_recon += recon.item()

    avg_loss  = total_loss  / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    print(f"VQ  [{epoch+1:3d}/{EPOCHS_VQ}]  total={avg_loss:.6f}  recon={avg_recon:.6f}")

    if (epoch + 1) % 10 == 0:
        torch.save(vq_model.state_dict(), CHECKPOINT_DIR / f"vq_epoch_{epoch+1}.pt")
    if avg_loss < best_vq_loss:
        best_vq_loss = avg_loss
        torch.save(vq_model.state_dict(), CHECKPOINT_DIR / "vq_best.pt")

torch.save(vq_model.state_dict(), CHECKPOINT_DIR / "vq_final.pt")
print(f"VQ-VAE done.  Best loss: {best_vq_loss:.6f}")


# ======================================
#  STAGE 2 — PREDICTOR TRAINING  (VQ frozen)
# ======================================
print("\n========== Stage 2: Predictor Training ==========")

for p in vq_model.parameters():
    p.requires_grad = False
vq_model.eval()

optimizer_pred = optim.AdamW(generator.parameters(), lr=LR_PRED)
best_pred_loss = float('inf')

for epoch in range(EPOCHS_PRED):
    generator.train()
    total_loss = 0.0

    for batch in tqdm(train_loader, desc=f"Pred {epoch+1}/{EPOCHS_PRED}", leave=False):
        vq_faces        = batch["vq_faces"].to(DEVICE)         # (B, 32, 56)
        speaker_full    = batch["speaker_full"].to(DEVICE)     # (B, 40, 56)
        audio_full      = batch["audio_full"].to(DEVICE)       # (B, 160, 128)
        listener_past   = batch["listener_past"].to(DEVICE)    # (B, 4)   long
        listener_future = batch["listener_future"].to(DEVICE)  # (B, 4)   long

        # Build the inputs dict that FACTModel.forward() expects
        inputs = {
            "listener_past" : listener_past,   # (B, 4)       token indices
            "speaker_full"  : speaker_full,    # (B, 40, 56)  motion
            "audio_full"    : audio_full,      # (B, 160, 128) audio
        }

        # FACTModel.forward(inputs, max_mask, mask_index)
        # max_mask=4 (cross_modal max_mask_len), mask_index=-1 (random masking)
        prediction = generator(inputs,
                               max_mask=max_mask_len,
                               mask_index=-1)
        # prediction: (B, sequence_length=9, n_embed=200)

        # Cross-entropy loss: prediction vs listener future token indices
        # Cut to min length for safety
        cut  = min(prediction.shape[1], listener_future.shape[1])
        loss = calc_logit_loss(prediction[:, :cut, :],
                               listener_future[:, :cut])

        optimizer_pred.zero_grad()
        loss.backward()
        optimizer_pred.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Pred [{epoch+1:3d}/{EPOCHS_PRED}]  loss={avg_loss:.6f}")

    if (epoch + 1) % 5 == 0:
        torch.save(generator.state_dict(), CHECKPOINT_DIR / f"pred_epoch_{epoch+1}.pt")
    if avg_loss < best_pred_loss:
        best_pred_loss = avg_loss
        torch.save(generator.state_dict(), CHECKPOINT_DIR / "pred_best.pt")

torch.save(generator.state_dict(), CHECKPOINT_DIR / "pred_final.pt")
print(f"Predictor done.  Best loss: {best_pred_loss:.6f}")

# ======================================
#  SUMMARY
# ======================================
print("\n========== Training Complete ==========")
print(f"  Checkpoints    : {CHECKPOINT_DIR}")
print(f"  VQ  best loss  : {best_vq_loss:.6f}")
print(f"  Pred best loss : {best_pred_loss:.6f}")
for f in ["vq_best.pt", "vq_final.pt", "pred_best.pt", "pred_final.pt"]:
    print(f"    {CHECKPOINT_DIR}/{f}")