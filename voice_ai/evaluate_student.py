import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torchaudio
import json
from pathlib import Path
from transformers import Wav2Vec2FeatureExtractor, HubertModel
import numpy as np
from tqdm import tqdm
from voice_ai.data import get_score_map
from voice_ai.models.student import DeepStudentNet, SimpleStudentNet, StatsStudentNet
from voice_ai.paths import OUTPUTS_DIR, get_dataset_dir
import random
from collections import defaultdict



student_size = 64
recording_type = "number"
dataset_name = "ukl_wavs"
epoch_to_load = "30"


dataset_path = str(get_dataset_dir(dataset_name, recording_type, split="healthy"))

model_type = "simple"
method_name = "student_learner"

words = ['happy', 'silly', 'funny', 'blue', 'red', 'green', 'fast', 'slow', 'bright', 'dark', 'dog', 'cat', 'tree', 'sky', 'sun', 'moon']

filter_rbhs = False

# ------------------------------
# Same StudentNet you trained
# ------------------------------
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


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def build_recording_scores_df(reconstruction_map):
    rows = []
    for recording_id, score in reconstruction_map.items():
        rows.append({
            "recording_id": recording_id,
            "patient_id": extract_patient_id(recording_id),
            "condition": extract_condition(recording_id),
            "mahalanobis_score": float(score),
        })
    return pd.DataFrame(rows)


def build_patient_scores_df(reconstruction_map, rbh_map):
    patient_rows = []
    patient_losses = defaultdict(list)
    patient_conditions = {}

    for recording_id, score in reconstruction_map.items():
        patient_id = extract_patient_id(recording_id)
        patient_losses[patient_id].append(score)
        patient_conditions[patient_id] = extract_condition(recording_id)

    for patient_id, losses in patient_losses.items():
        row = {
            "patient_id": patient_id,
            "condition": patient_conditions.get(patient_id),
            "mean_mahalanobis_score": float(np.mean(losses)),
            "num_recordings": int(len(losses)),
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
        & patient_scores_df["R"].notna()
        & patient_scores_df["B"].notna()
        & patient_scores_df["H"].notna()
    ].copy()


def results_base_dir_for_model(current_model_type):
    random_folder_name = "_".join(random.choices(words, k=3))
    return str(OUTPUTS_DIR / dataset_name / "paper" / method_name / current_model_type / random_folder_name)


def checkpoint_path_for_epoch(epoch_value, current_model_type):
    return f"checkpoints_{dataset_name}/student_{recording_type}_{dataset_name}_{current_model_type}_epoch{epoch_value}.pt"


class SnippetDataset(torch.utils.data.Dataset):
    def __init__(self, wav_dir, snippet_seconds=1.0, sample_rate=16000):
        self.paths = sorted([p for p in Path(wav_dir).rglob("*.wav")])

        if dataset_name == "ukl_wavs" and filter_rbhs:
            self.paths = [p for p in self.paths if p.stem.rsplit("_",2)[0] in rbh_map and rbh_map[p.stem.rsplit("_",2)[0]][3] <= 2.5]

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

        T = wav.shape[0]
        N = self.target_samples

        if T == N:
            return wav

        if T < N:
            pad = N - T
            wav = F.pad(wav, (0, pad))
            return wav

        start = (T - N) // 2
        return wav[start:start + N]


# ----------------------------------------
# Teacher helper (same as training)
# ----------------------------------------

@torch.no_grad()
def get_teacher_repr(wavs, teacher, processor, device):
    if wavs.ndim == 3:
        wavs = wavs.squeeze(1)

    inputs = processor(
        wavs.detach().cpu().numpy(),
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = teacher(**inputs)
    hs = outputs.last_hidden_state
    hs = hs.mean(dim=1)

    return hs


# ----------------------------------------
# Evaluation function
# ----------------------------------------

def evaluate_folder(
    eval_dir,
    healthy_dir,
    checkpoint_path,
    cosine_weight=0.5,
    target_seconds=1.0,
    sample_rate=16000,
    device="cuda:6",
    model_type="simple",
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print(f"Loading student model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Load teacher
    trainer_name = "facebook/hubert-base-ls960"
    print(f"Loading teacher model from {trainer_name}...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(trainer_name)
    teacher = HubertModel.from_pretrained(trainer_name).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    emb_dim = 768

    # Load student
    if model_type == "simple":
        student = SimpleStudentNet(out_dim=emb_dim, sample_rate=sample_rate, layer_size=student_size).to(device)
    elif model_type == "deep":
        student = DeepStudentNet(out_dim=emb_dim, sample_rate=sample_rate).to(device)
    else:
        student = StatsStudentNet(out_dim=emb_dim, sample_rate=sample_rate).to(device)
    student.load_state_dict(ckpt["model"])
    student.eval()

    # --- Calculate Mean and Covariance from Healthy Training Data ---
    print("Calculating statistics on healthy training data...")
    healthy_dataset = SnippetDataset(healthy_dir, snippet_seconds=target_seconds, sample_rate=sample_rate)
    healthy_loader = torch.utils.data.DataLoader(healthy_dataset, batch_size=32, shuffle=False)

    error_vectors = []
    with torch.no_grad():
        for wavs in tqdm(healthy_loader, desc="Processing healthy samples"):
            wavs = wavs.to(device)
            teacher_emb = get_teacher_repr(wavs, teacher, processor, device)
            student_emb = student(wavs)
            error = teacher_emb - student_emb
            error_vectors.append(error.cpu())

    error_vectors = torch.cat(error_vectors, dim=0)
    
    # Add a small regularization term to the covariance matrix for stability
    mean_error = error_vectors.mean(dim=0).to(device)
    cov = torch.cov(error_vectors.T)
    reg_eps = 1e-5
    reg = torch.eye(cov.shape[0]) * reg_eps
    inv_cov = torch.inverse(cov + reg).to(device)

    centered_errors = error_vectors - error_vectors.mean(dim=0, keepdim=True)
    inv_cov_cpu = inv_cov.cpu()
    healthy_distances = torch.sqrt(
        torch.einsum("bi,ij,bj->b", centered_errors, inv_cov_cpu, centered_errors).clamp_min(0.0)
    )

    healthy_stats = {
        "healthy_num_samples": int(error_vectors.shape[0]),
        "embedding_dim": int(error_vectors.shape[1]),
        "covariance_regularization": reg_eps,
        "mean_error_l2_norm": float(mean_error.norm().item()),
        "healthy_distance_summary": summarize_scores(healthy_distances.tolist()),
        "trainer_name": trainer_name,
    }

    print("Finished calculating statistics.")

    results = {}

    wav_paths = sorted(Path(eval_dir).rglob("*.wav"))
    if not wav_paths:
        raise ValueError(f"No wav files found under {eval_dir}")

    for path in tqdm(wav_paths, desc="Evaluating RBH samples"):
        wav, sr = torchaudio.load(path)
        wav = wav.mean(dim=0)

        if sr != sample_rate:
            wav = torchaudio.functional.resample(wav, sr, sample_rate)

        T = wav.shape[0]
        N = int(target_seconds * sample_rate)

        if T < N:
            wav = F.pad(wav, (0, N - T))
        elif T > N:
            start = (T - N) // 2
            wav = wav[start:start + N]

        wav = wav.unsqueeze(0).to(device)  # (1, T)

        with torch.no_grad():
            teacher_emb = get_teacher_repr(wav, teacher, processor, device)
            student_emb = student(wav)

            # Calculate Mahalanobis distance
            error_vec = teacher_emb - student_emb
            diff = error_vec - mean_error
            dist = torch.sqrt(torch.dot(diff.squeeze(), torch.matmul(inv_cov, diff.squeeze())))

        pth_cleaned = normalize_recording_key(path.stem)
        # print(pth_cleaned)
        results[pth_cleaned] = float(dist.item())

    healthy_stats["eval_num_recordings"] = int(len(results))

    return results, healthy_stats


def build_evaluation_cache_payload(reconstruction_map, healthy_stats, metadata):
    return {
        "metadata": metadata,
        "reconstruction_map": {str(k): float(v) for k, v in reconstruction_map.items()},
        "healthy_stats": healthy_stats,
    }


def resolve_cache_path(plot_dir: Path, cache_path_arg: str):
    if cache_path_arg:
        return Path(cache_path_arg)
    return plot_dir / "evaluation_cache.json"


def load_evaluation_cache(cache_path: Path):
    if not cache_path.is_file():
        return None
    payload = load_json(cache_path)
    reconstruction_map = {
        str(k): float(v) for k, v in payload.get("reconstruction_map", {}).items()
    }
    healthy_stats = payload.get("healthy_stats", {})
    metadata = payload.get("metadata", {})
    if not reconstruction_map or not healthy_stats:
        return None
    return reconstruction_map, healthy_stats, metadata


def cache_metadata_matches(actual_metadata, cached_metadata):
    keys_to_compare = [
        "checkpoint_path",
        "healthy_dir",
        "eval_dir",
        "target_seconds",
        "device",
        "model_type",
        "recording_type",
        "dataset_name",
    ]
    for key in keys_to_compare:
        if actual_metadata.get(key) != cached_metadata.get(key):
            return False
    return True


# ----------------------------------------
# CLI entrypoints
# ----------------------------------------

def run_single_evaluation(args):
    """Score the evaluation recordings, write the score cache and score tables, and plot Figure 2."""
    selected_model_type = args.model_type
    checkpoint_path = args.checkpoint_path if args.checkpoint_path else checkpoint_path_for_epoch(args.epoch_to_load, selected_model_type)
    plot_dir = Path(f"{results_base_dir_for_model(selected_model_type)}/correlation_plots_{recording_type}/")
    plot_dir.mkdir(parents=True, exist_ok=True)
    cache_path = resolve_cache_path(plot_dir, args.cache_path)
    evaluation_metadata = {
        "checkpoint_path": str(checkpoint_path),
        "healthy_dir": str(args.healthy_dir),
        "eval_dir": str(args.eval_dir),
        "target_seconds": float(args.target_seconds),
        "device": str(args.device),
        "model_type": str(selected_model_type),
        "recording_type": str(recording_type),
        "dataset_name": str(dataset_name),
    }

    loaded_cache = load_evaluation_cache(cache_path)
    use_cached_evaluation = False
    if args.cache_mode in {"auto", "load"} and loaded_cache is not None:
        cached_reconstruction_map, cached_healthy_stats, cached_metadata = loaded_cache
        if cache_metadata_matches(evaluation_metadata, cached_metadata):
            reconstruction_map = cached_reconstruction_map
            healthy_stats = cached_healthy_stats
            use_cached_evaluation = True
            print(f"Loaded cached evaluation from {cache_path}")
        elif args.cache_mode == "load":
            raise ValueError(f"Cache at {cache_path} does not match the current evaluation settings.")

    if not use_cached_evaluation:
        reconstruction_map, healthy_stats = evaluate_folder(
            healthy_dir=args.healthy_dir,
            eval_dir=args.eval_dir,
            checkpoint_path=checkpoint_path,
            cosine_weight=0.5,
            target_seconds=args.target_seconds,
            device=args.device,
            model_type=selected_model_type,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_json(cache_path, build_evaluation_cache_payload(reconstruction_map, healthy_stats, evaluation_metadata))
        print(f"Saved evaluation cache to {cache_path}")

    _, rbh_map, _ = get_score_map()
    recording_scores_df = build_recording_scores_df(reconstruction_map)
    recording_scores_df.to_csv(plot_dir / "recording_scores.csv", index=False)
    patient_scores_df = build_patient_scores_df(reconstruction_map, rbh_map)
    patient_scores_df.to_csv(plot_dir / "patient_scores.csv", index=False)
    patient_scores_filtered = filter_patient_scores_for_rbh(patient_scores_df)

    # Figure 2: patient-level scores by perceptual rating
    losses_list = patient_scores_filtered["mean_mahalanobis_score"].tolist()
    for name in ["R", "B", "H"]:
        df = pd.DataFrame({"Reconstruction Loss": losses_list, f"{name} Score": patient_scores_filtered[name].tolist()})
        if df.empty:
            print(f"Not enough data for {name} boxplot.")
            continue
        score_categories = sorted(df[f"{name} Score"].unique())
        df[f"{name} Score"] = pd.Categorical(df[f"{name} Score"], categories=score_categories, ordered=True)

        plt.figure(figsize=(10, 8))
        sns.boxplot(x="Reconstruction Loss", y=f"{name} Score", data=df, orient="h", whis=[0, 100], width=.6, palette="vlag")
        sns.stripplot(x="Reconstruction Loss", y=f"{name} Score", data=df, orient="h", color=".25", alpha=0.5)
        plt.title(f"Mahalanobis Score by {name} Rating")
        plt.xlabel("Mahalanobis Score")
        plt.ylabel(f"{name} Score")
        plt.xlim(df["Reconstruction Loss"].quantile(.05), df["Reconstruction Loss"].quantile(.95))
        plt.grid(True, linestyle="--", alpha=0.6)
        plot_path = plot_dir / f"reconstruction_loss_vs_{name.lower()}_boxplot.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved {name} boxplot to {plot_path}")

    save_json(plot_dir / "run_summary.json", {
        "checkpoint_path": str(checkpoint_path),
        "cache_path": str(cache_path),
        "used_cached_evaluation": bool(use_cached_evaluation),
        "recording_score_summary": summarize_scores(recording_scores_df["mahalanobis_score"].tolist()),
        "patient_score_summary": summarize_scores(patient_scores_df["mean_mahalanobis_score"].tolist()),
        "healthy_reference_stats": healthy_stats,
        "num_recordings_scored": int(len(recording_scores_df)),
        "num_patients_with_rbh": int(len(patient_scores_filtered)),
    })


def parse_args():
    parser = argparse.ArgumentParser(description="Score recordings with a trained student checkpoint.")
    parser.add_argument("--model-type", choices=["simple", "deep", "stats"], default=model_type)
    parser.add_argument("--epoch-to-load", type=int, default=int(epoch_to_load))
    parser.add_argument("--checkpoint-path", type=str, default="")
    parser.add_argument("--target-seconds", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-dir", type=Path, default=get_dataset_dir(dataset_name, recording_type, split="unhealthy"))
    parser.add_argument("--healthy-dir", type=Path, default=Path(dataset_path))
    parser.add_argument("--cache-mode", choices=["auto", "recompute", "load"], default="auto")
    parser.add_argument("--cache-path", type=str, default="")
    return parser.parse_args()


if __name__ == "__main__":
    run_single_evaluation(parse_args())
