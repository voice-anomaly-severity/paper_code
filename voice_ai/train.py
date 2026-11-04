#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset, DataLoader
from transformers import Wav2Vec2FeatureExtractor, HubertModel
from tqdm import tqdm
from voice_ai.data import get_score_map
from voice_ai.models.student import DeepStudentNet, SimpleStudentNet, StatsStudentNet
from voice_ai.paths import CHECKPOINTS_DIR, get_dataset_dir

_, rbh_map, _ = get_score_map()

recording_type = "number"
# dataset_name = "svd"
# dataset_name = "ukl_patho_wavs"
dataset_name = "ukl_wavs"
# dataset_name = "life"

rbh_filter = False


class SnippetDataset(Dataset):
    def __init__(self, wav_dir, snippet_seconds=1.0, sample_rate=16000, split_chunks=False):
        self.paths = sorted([p for p in Path(wav_dir).rglob("*.wav")])
        
        # filter rbhs

        print(f"reading data from{wav_dir}")

        all_paths = len(self.paths)

        if dataset_name == "ukl_wavs" and rbh_filter:
            self.paths = [p for p in self.paths if p.stem.rsplit("_",2)[0] in rbh_map and rbh_map[p.stem.rsplit("_",2)[0]][3] <= 2.5]

        print(f"Filtered from {all_paths} to {len(self.paths)} files based on RBH scores")
        if not self.paths:
            raise ValueError(f"No .wav files found in {wav_dir}")

        self.sample_rate = sample_rate
        self.target_samples = int(snippet_seconds * sample_rate)
        self.split_chunks = split_chunks
        self.indices = []

        if self.split_chunks:
            print(f"Scanning {len(self.paths)} files to create chunks...")
            for i, p in enumerate(tqdm(self.paths)):
                try:
                    info = torchaudio.info(p)
                    total_samples = int(info.num_frames * self.sample_rate / info.sample_rate)
                    
                    if total_samples < self.target_samples:
                        self.indices.append((i, -1))
                    else:
                        n_chunks = total_samples // self.target_samples
                        for c in range(n_chunks):
                            self.indices.append((i, c))
                except Exception as e:
                    print(f"Error processing {p}: {e}")
        else:
            self.indices = [(i, -1) for i in range(len(self.paths))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path_idx, chunk_idx = self.indices[idx]
        path = self.paths[path_idx]
        wav, sr = torchaudio.load(path)

        # convert to mono
        wav = wav.mean(dim=0)

        # resample if needed
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        T = wav.shape[0]
        N = self.target_samples

        if T == N:
            return wav

        if T < N:
            # pad at the end
            pad = N - T
            wav = F.pad(wav, (0, pad))
            return wav

        # T > N
        if chunk_idx == -1:
            start = random.randint(0, T - N)
        else:
            start = chunk_idx * N
            if start + N > T: start = T - N
            
        return wav[start:start + N]


# ---------------------------
# Dataset with random snippets
# ---------------------------

class RandomSnippetDataset(Dataset):
    """
    Each item is a random snippet (fixed duration) from a full-length wav file.
    No pre-cutting: snippets are sampled on the fly each __getitem__ call.
    """

    def __init__(self, wav_dir, snippet_seconds=2.0, sample_rate=16000):
        self.paths = sorted(
            [p for p in Path(wav_dir).rglob("*.wav")]
        )
        if len(self.paths) == 0:
            raise ValueError(f"No .wav files found in {wav_dir}")
        self.snippet_samples = int(snippet_seconds * sample_rate)
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav, sr = torchaudio.load(path)

        # Resample to target sr
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        # Mono
        wav = wav.mean(dim=0)  # (T,)

        T = wav.shape[0]
        N = self.snippet_samples

        if T <= N:
            # Pad if file shorter than snippet
            pad = N - T
            wav = F.pad(wav, (0, pad))
            return wav

        # Random crop
        start = random.randint(0, T - N)
        end = start + N
        snippet = wav[start:end]
        return snippet


def collate_fn(batch):
    # All snippets already same length
    return torch.stack(batch, dim=0)  # (B, T)


# ---------------------------
# Student network
# ---------------------------

# ---------------------------
# Teacher utility
# ---------------------------

@torch.no_grad()
def get_teacher_repr(wavs, teacher, processor, device):
    """
    wavs: (B, T) on device
    Returns: (B, H) snippet-level embedding (mean pooled over time)
    """
    # processor can handle torch tensors (B, T)
    if wavs.ndim == 3:  # (B, 1, T)
        wavs = wavs.squeeze(1)

    inputs = processor(
        wavs.detach().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True
    )
    # Ensure inputs on device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Request hidden states and select an intermediate one
    outputs = teacher(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    # hidden_states is a tuple of (embedding_output, layer_1_output, ..., layer_n_output)
    # We'll pick a layer from the middle, e.g., layer 6 for hubert-base (12 layers total)
    hs = outputs.last_hidden_state
    # if "attention_mask" in inputs:
    #     mask = inputs["attention_mask"].unsqueeze(-1)  # (B, T, 1)
    #     hs = (hs * mask).sum(dim=1) / mask.sum(dim=1)
    # else:
    hs = hs.mean(dim=1)

    return hs  # (B, H)


# ---------------------------
# Training loop
# ---------------------------

def train(args):
    # Repro
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Dataset
    dataset = SnippetDataset(
        wav_dir=args.wav_dir,
        snippet_seconds=args.snippet_seconds,
        sample_rate=16000,
        split_chunks=(dataset_name == "svd"),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=True,
    )
    print(f"Found {len(dataset)} files")

    # Teacher
    teacher_name = "facebook/hubert-base-ls960"
    print(f"Loading teacher: {teacher_name}")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(teacher_name)
    teacher = HubertModel.from_pretrained(teacher_name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    teacher_dim = 768

    print(f"Teacher embedding dim: {teacher_dim}")

    # Student
    if args.model_type == "simple":
        student = SimpleStudentNet(out_dim=teacher_dim, sample_rate=16000).to(device)
    elif args.model_type == "deep":
        student = DeepStudentNet(out_dim=teacher_dim, sample_rate=16000).to(device)
    else:
        student = StatsStudentNet(out_dim=teacher_dim, sample_rate=16000).to(device)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Optionally resume
    if args.load_student and Path(args.load_student).is_file():
        print(f"Loading student weights from {args.load_student}")
        ckpt = torch.load(args.load_student, map_location=device)
        student.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])

    # Training
    student.train()
    global_step = 0

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for wavs in pbar:
            wavs = wavs.to(device)

            with torch.no_grad():
                teacher_repr = get_teacher_repr(wavs, teacher, processor, device)  # (B, H)

            student_repr = student(wavs)  # (B, H)

            mse = F.mse_loss(student_repr, teacher_repr)
            cos = 1.0 - F.cosine_similarity(student_repr, teacher_repr, dim=-1).mean()
            loss = mse + args.cosine_weight * cos

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % args.log_every == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "mse": f"{mse.item():.4f}",
                    "cos": f"{cos.item():.4f}",
                })

        avg_loss = epoch_loss / max(1, n_batches)
        print(f"Epoch {epoch+1}: avg_loss={avg_loss:.4f}")

        # Save checkpoint
        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        ckpt_path = out_path / f"student_{recording_type}_{dataset_name}_{args.model_type}_epoch{epoch+1}.pt"
        torch.save(
            {
                "model": student.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "global_step": global_step,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"Saved checkpoint to {ckpt_path}")


# ---------------------------
# CLI
# ---------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Distill HuBERT representations on healthy voice snippets"
    )

    dataset_path = get_dataset_dir(dataset_name, recording_type, split="healthy")

    parser.add_argument("--wav_dir", type=Path, default=dataset_path)
    parser.add_argument("--out_dir", type=Path, default=CHECKPOINTS_DIR / dataset_name,
                        help="Where to store student checkpoints.")
    parser.add_argument("--model_type", type=str, default="simple", choices=["simple", "deep", "stats"],
                        help="Which student model architecture to use.")
    parser.add_argument("--snippet_seconds", type=float, default=1.0,
                        help="Length of random snippets in seconds.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--cosine_weight", type=float, default=0.5,
                        help="Weight for cosine loss term.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU even if CUDA is available.")
    parser.add_argument("--load_student", type=str, default="",
                        help="Optional path to existing student checkpoint (for resume).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
