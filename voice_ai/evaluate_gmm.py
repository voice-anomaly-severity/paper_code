import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from voice_ai.data import get_score_map
from voice_ai.paths import OUTPUTS_DIR, get_dataset_dir


method_name = "gmm"
dataset_name = "ukl_wavs"
recording_type = "number"
trainer_name = "facebook/hubert-base-ls960"
words = [
    "happy",
    "silly",
    "funny",
    "blue",
    "red",
    "green",
    "fast",
    "slow",
    "bright",
    "dark",
    "dog",
    "cat",
    "tree",
    "sky",
    "sun",
    "moon",
]

_, rbh_map, _ = get_score_map()


def normalize_recording_key(key: str) -> str:
    return key


def extract_patient_id(key: str) -> str:
    normalized = normalize_recording_key(key)
    parts = normalized.rsplit("_", 2)
    if len(parts) == 3:
        return parts[0]
    return normalized


def extract_condition(key: str) -> str:
    normalized = normalize_recording_key(key)
    return normalized.split("_", 1)[0]


def save_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def summarize_scores(values):
    if not values:
        return {}

    arr = np.asarray(values, dtype=float)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p05": float(np.quantile(arr, 0.05)),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(arr.max()),
    }


def build_recording_scores_df(score_map):
    rows = []
    for recording_id, score in score_map.items():
        rows.append(
            {
                "recording_id": recording_id,
                "patient_id": extract_patient_id(recording_id),
                "condition": extract_condition(recording_id),
                "gmm_score": float(score),
            }
        )
    return pd.DataFrame(rows)


def build_patient_scores_df(score_map, rbh_map):
    patient_rows = []
    patient_scores = defaultdict(list)
    patient_conditions = {}

    for recording_id, score in score_map.items():
        patient_id = extract_patient_id(recording_id)
        patient_scores[patient_id].append(score)
        patient_conditions[patient_id] = extract_condition(recording_id)

    for patient_id, scores in patient_scores.items():
        row = {
            "patient_id": patient_id,
            "condition": patient_conditions.get(patient_id),
            "mean_gmm_score": float(np.mean(scores)),
            "num_recordings": int(len(scores)),
        }
        if patient_id in rbh_map:
            row["RBH"] = float(rbh_map[patient_id][0])
            row["R"] = float(rbh_map[patient_id][1])
            row["B"] = float(rbh_map[patient_id][2])
            row["H"] = float(rbh_map[patient_id][3])
        patient_rows.append(row)

    return pd.DataFrame(patient_rows)


def filter_patient_scores_for_rbh(patient_scores_df):
    return patient_scores_df[
        (~patient_scores_df["patient_id"].astype(str).str.contains("Phon_GA", na=False))
        & patient_scores_df["RBH"].notna()
    ].copy()


def results_base_dir():
    random_folder_name = "_".join(random.choices(words, k=3))
    return OUTPUTS_DIR / dataset_name / "paper" / method_name / random_folder_name


class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir, snippet_seconds=1.0, sample_rate=16000):
        self.paths = sorted([p for p in Path(wav_dir).rglob("*.wav")])
        if not self.paths:
            raise ValueError(f"No .wav files found in {wav_dir}")
        self.sample_rate = sample_rate
        self.target_samples = int(snippet_seconds * sample_rate)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)

        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)

        target = self.target_samples
        if wav.shape[0] < target:
            wav = F.pad(wav, (0, target - wav.shape[0]))
        elif wav.shape[0] > target:
            start = (wav.shape[0] - target) // 2
            wav = wav[start : start + target]

        return wav


@torch.no_grad()
def get_teacher_repr(wavs, teacher, processor, device):
    if wavs.ndim == 3:
        wavs = wavs.squeeze(1)

    inputs = processor(
        wavs,
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    input_values = inputs["input_values"].to(device).squeeze(0)
    outputs = teacher(input_values=input_values, attention_mask=None)
    return outputs.last_hidden_state.mean(dim=1)


def fit_gmm_on_healthy_embeddings(healthy_dir, target_seconds, sample_rate, device, n_components, covariance_type):
    processor = Wav2Vec2FeatureExtractor.from_pretrained(trainer_name, local_files_only=True)
    teacher = HubertModel.from_pretrained(trainer_name, local_files_only=True).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    dataset = SnippetDataset(healthy_dir, snippet_seconds=target_seconds, sample_rate=sample_rate)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    all_embeddings = []
    for wavs in tqdm(loader, desc="Processing healthy samples"):
        wavs = wavs.to(device)
        embeddings = get_teacher_repr(wavs, teacher, processor, device)
        all_embeddings.append(embeddings.cpu().numpy())

    embeddings_array = np.concatenate(all_embeddings, axis=0)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=42,
        verbose=1,
        verbose_interval=10,
    )
    gmm.fit(embeddings_array)

    healthy_log_likelihood = gmm.score_samples(embeddings_array)
    healthy_stats = {
        "healthy_num_samples": int(embeddings_array.shape[0]),
        "embedding_dim": int(embeddings_array.shape[1]),
        "n_components": int(n_components),
        "covariance_type": covariance_type,
        "converged": bool(gmm.converged_),
        "n_iter": int(gmm.n_iter_),
        "healthy_negative_log_likelihood_summary": summarize_scores((-healthy_log_likelihood).tolist()),
    }

    return gmm, teacher, processor, healthy_stats


def score_eval_folder(eval_dir, teacher, processor, gmm, target_seconds, sample_rate, device):
    results = {}
    wav_paths = sorted(Path(eval_dir).rglob("*.wav"))
    if not wav_paths:
        raise ValueError(f"No wav files found under {eval_dir}")

    target = int(target_seconds * sample_rate)
    for path in tqdm(wav_paths, desc="Evaluating samples"):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)

        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        if wav.shape[0] < target:
            wav = F.pad(wav, (0, target - wav.shape[0]))
        elif wav.shape[0] > target:
            start = (wav.shape[0] - target) // 2
            wav = wav[start : start + target]

        wav = wav.unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = get_teacher_repr(wav, teacher, processor, device).cpu().numpy()
        score = -gmm.score_samples(embedding)[0]
        results[normalize_recording_key(path.stem)] = float(score)

    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a GMM baseline on frozen HuBERT embeddings.")
    parser.add_argument("--healthy-dir", type=Path, default=get_dataset_dir(dataset_name, recording_type, split="healthy"))
    parser.add_argument("--eval-dir", type=Path, default=get_dataset_dir(dataset_name, recording_type, split="unhealthy"))
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--covariance-type", type=str, default="full", choices=["full", "tied", "diag", "spherical"])
    return parser.parse_args()


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    plot_dir = results_base_dir() / f"correlation_plots_{recording_type}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    gmm, teacher, processor, healthy_stats = fit_gmm_on_healthy_embeddings(
        healthy_dir=args.healthy_dir,
        target_seconds=args.target_seconds,
        sample_rate=args.sample_rate,
        device=device,
        n_components=args.n_components,
        covariance_type=args.covariance_type,
    )

    score_map = score_eval_folder(
        eval_dir=args.eval_dir,
        teacher=teacher,
        processor=processor,
        gmm=gmm,
        target_seconds=args.target_seconds,
        sample_rate=args.sample_rate,
        device=device,
    )

    recording_scores_df = build_recording_scores_df(score_map)
    recording_scores_df.to_csv(plot_dir / "recording_scores.csv", index=False)

    patient_scores_df = build_patient_scores_df(score_map, rbh_map)
    patient_scores_df.to_csv(plot_dir / "patient_scores.csv", index=False)

    patient_scores_filtered = filter_patient_scores_for_rbh(patient_scores_df)

    run_summary = {
        "dataset_name": dataset_name,
        "recording_type": recording_type,
        "method_name": method_name,
        "trainer_name": trainer_name,
        "healthy_dir": str(args.healthy_dir),
        "eval_dir": str(args.eval_dir),
        "target_seconds": args.target_seconds,
        "sample_rate": args.sample_rate,
        "n_components": args.n_components,
        "covariance_type": args.covariance_type,
        "recording_score_summary": summarize_scores(recording_scores_df["gmm_score"].tolist()),
        "patient_score_summary": summarize_scores(patient_scores_df["mean_gmm_score"].tolist()),
        "healthy_reference_stats": healthy_stats,
        "num_recordings_scored": int(len(recording_scores_df)),
        "num_patients_scored": int(len(patient_scores_df)),
        "num_patients_with_rbh": int(len(patient_scores_filtered)),
    }

    save_json(plot_dir / "run_summary.json", run_summary)


if __name__ == "__main__":
    main(parse_args())
