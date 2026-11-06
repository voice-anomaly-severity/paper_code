"""Cross-fitting by speaker: train the student on the healthy speakers of all but one fold and score every clip.

Training and scoring follow voice_ai/train.py and voice_ai/evaluate_student.py (SimpleStudentNet, 64 channels,
HuBERT-base teacher, 1-s snippets, batch 16, lr 1e-4, cosine weight 0.5, seed 42, epoch 30). The reference mean
and covariance are estimated on the training words only.

    python run_fold.py --healthy-clips <dir> --patient-clips <dir> --out runs --fold 0 --folds 5 --partition 0
    python run_fold.py ... --fold -1                          # all healthy speakers (setup of the main model)
    python run_fold.py ... --fold -1 --checkpoint <epoch-30>  # score with an existing checkpoint, no training

Clip names are <recording>_<word>_<index>.wav (segmentation/extract_number_words.py). Healthy recordings
are those whose name starts with Phon_GA; their speakers are identified by a hash of the identity columns of the
clinical workbook, computed in memory only. Writes runs/scores_p<partition>_fold<fold>.csv (one row per clip).
"""
import argparse
import glob
import hashlib
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from transformers import HubertModel, Wav2Vec2FeatureExtractor

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from voice_ai.models.student import SimpleStudentNet  # noqa: E402

DATA_ROOT = Path(os.environ.get('VOICE_DATA_ROOT') or REPO)
EPOCHS, BATCH, LR, COSW, SEED, N = 30, 16, 1e-4, 0.5, 42, 16000
HEALTHY_MARKER = 'Phon_GA'


def rec_of(path):
    """Recording ID of a clip: <recording>_<word>_<index>.wav (also accepts <recording>__<index>_<word>.wav)."""
    name = os.path.basename(path)[:-4]
    return name.split('__')[0] if '__' in name else name.rsplit('_', 2)[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--healthy-clips', required=True)
    ap.add_argument('--patient-clips', required=True)
    ap.add_argument('--out', default='runs')
    ap.add_argument('--fold', type=int, required=True, help='held-out fold, or -1 for all healthy speakers')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--partition', type=int, default=0, help='seed of the split of healthy speakers into folds')
    ap.add_argument('--workbook', type=Path, default=DATA_ROOT / 'Voice_KI_RBH_full.xlsx')
    ap.add_argument('--checkpoint', type=Path, help='score with this student checkpoint instead of training')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    paths = sorted(glob.glob(os.path.join(args.healthy_clips, '*.wav'))) + sorted(
        p for p in glob.glob(os.path.join(args.patient_clips, '*.wav')) if not os.path.basename(p).startswith(HEALTHY_MARKER))
    healthy = [p for p in paths if rec_of(p).startswith(HEALTHY_MARKER)]

    frame = pd.read_excel(args.workbook, sheet_name='Phon-Gutachten')
    person = {}
    for _, r in frame[frame['Dateiname'].notna()].iterrows():
        key = '|'.join(str(r[c]).strip().lower() for c in ['Nachname', 'Vorname', 'Geburtsdatum'])
        person[str(r['Dateiname']).replace('.wav', '').strip()] = hashlib.sha256(key.encode()).hexdigest()
    recs = sorted({rec_of(p) for p in healthy})
    pid = {r: person.get(r, 'unmatched_' + r) for r in recs}
    people = sorted(set(pid.values()))
    random.Random(args.partition).shuffle(people)
    fold_of_person = {q: i % args.folds for i, q in enumerate(people)}
    fold_of_rec = {r: fold_of_person[pid[r]] for r in recs}
    train = [p for p in healthy if args.fold < 0 or fold_of_rec[rec_of(p)] != args.fold]
    print('healthy recordings', len(recs), 'people', len(people), flush=True)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(4)
    cache = os.path.join(args.out, 'waves16k.pt')          # resample once, shared by all folds
    if os.path.exists(cache):
        waves = torch.load(cache)
    else:
        resamplers = {}

        def load(p):
            w, sr = torchaudio.load(p)
            w = w.mean(0)
            if sr == 16000:
                return w
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(sr, 16000)
            return resamplers[sr](w)
        waves = {p: load(p) for p in paths}
        torch.save(waves, cache)
    waves = {p: waves[p] for p in paths}

    def random_crop(w):
        T = w.shape[0]
        if T <= N:
            return F.pad(w, (0, N - T))
        s = random.randint(0, T - N)
        return w[s:s + N]

    def center_crop(w):
        T = w.shape[0]
        if T <= N:
            return F.pad(w, (0, N - T))
        s = (T - N) // 2
        return w[s:s + N]

    proc = Wav2Vec2FeatureExtractor.from_pretrained('facebook/hubert-base-ls960')
    teacher = HubertModel.from_pretrained('facebook/hubert-base-ls960').to(dev).eval()
    for q in teacher.parameters():
        q.requires_grad = False

    @torch.no_grad()
    def teach(b):
        x = proc(b.cpu().numpy(), sampling_rate=16000, return_tensors='pt', padding=True)
        return teacher(**{k: v.to(dev) for k, v in x.items()}).last_hidden_state.mean(1)

    student = SimpleStudentNet(out_dim=768, sample_rate=16000, layer_size=64).to(dev)
    opt = torch.optim.AdamW(student.parameters(), lr=LR, weight_decay=0.0)
    student.train()
    epochs = EPOCHS
    if args.checkpoint:
        student.load_state_dict(torch.load(args.checkpoint, map_location=dev)['model'])
        epochs = 0
    for ep in range(epochs):
        order = train[:]
        random.shuffle(order)
        tot, nb = 0.0, 0
        for i in range(0, len(order) - BATCH + 1, BATCH):          # drop_last, as in the paper
            b = torch.stack([random_crop(waves[p]) for p in order[i:i + BATCH]]).to(dev)
            t, s = teach(b), student(b)
            loss = F.mse_loss(s, t) + COSW * (1 - F.cosine_similarity(s, t, dim=-1).mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            nb += 1
        print(f'fold {args.fold} epoch {ep + 1} loss {tot / nb:.4f}', flush=True)
    student.eval()

    @torch.no_grad()
    def errors(ps, crop):
        out = []
        for i in range(0, len(ps), 64):
            b = torch.stack([crop(waves[p]) for p in ps[i:i + 64]]).to(dev)
            out.append((teach(b) - student(b)).cpu())
        return torch.cat(out).double()

    # reference distribution: training words, random 1-s crops (as voice_ai/evaluate_student.py)
    E = errors(train, random_crop)
    mu = E.mean(0)
    inv = torch.inverse(torch.cov(E.T) + 1e-5 * torch.eye(E.shape[1], dtype=E.dtype))
    D = errors(paths, center_crop) - mu
    dist = torch.sqrt(torch.einsum('bi,ij,bj->b', D, inv, D).clamp_min(0)).numpy()

    rows = []
    for p, d in zip(paths, dist):
        r = rec_of(p)
        if r.startswith(HEALTHY_MARKER):
            role = 'healthy_train' if (args.fold < 0 or fold_of_rec[r] != args.fold) else 'healthy_heldout'
        else:
            role = 'patient'
        rows.append({'partition': args.partition, 'fold': args.fold, 'recording': r, 'clip': os.path.basename(p),
                     'role': role, 'person_fold': fold_of_rec.get(r, -1), 'score': float(d)})
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f'scores_p{args.partition}_fold{args.fold}.csv'), index=False)
    print('DONE', args.fold, len(train), 'training words', flush=True)


if __name__ == '__main__':
    main()
