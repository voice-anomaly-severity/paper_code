"""Cut recordings of the counting task into single number words with a German wav2vec2 model.

Used for the cross-fitted comparison with held-out healthy speakers (crossfit/). The model
jonatasgrosman/wav2vec2-large-xlsr-53-german transcribes each recording with word timestamps; every word
containing "undzwanzig" is saved as a 1-s clip (centred, zero-padded or trimmed) at the original sampling rate.
Scored with the epoch-30 checkpoint of the main model, these clips reproduce the reported recording scores
(crossfit/validate_resegmentation.py).

    python extract_number_words_w2v2.py --input-dir <healthy wavs> --output-dir clips/healthy
    python extract_number_words_w2v2.py --input-dir <patient wavs> --output-dir clips/patients --skip-a

Clip names: <recording>__<index>_<number word>.wav
"""
import argparse
import os
from pathlib import Path

import torch
import torchaudio
from rapidfuzz import process
from transformers import pipeline

TARGET_WORDS = ['zwanzig', 'einundzwanzig', 'zweiundzwanzig', 'dreiundzwanzig', 'vierundzwanzig', 'fünfundzwanzig',
                'sechsundzwanzig', 'siebenundzwanzig', 'achtundzwanzig', 'neunundzwanzig']


def one_second(waveform, sample_rate, start_s, end_s):
    segment = waveform[:, int(float(start_s) * sample_rate):int(float(end_s) * sample_rate)]
    target, current = int(sample_rate), segment.shape[1]
    if current < target:
        before = (target - current) // 2
        return torch.nn.functional.pad(segment, (before, target - current - before))
    if current > target:
        start = (current - target) // 2
        return segment[:, start:start + target]
    return segment


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--skip-a', action='store_true',
                    help="skip recordings whose name contains '_a' (done for the patient cohort)")
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    asr = pipeline('automatic-speech-recognition', model='jonatasgrosman/wav2vec2-large-xlsr-53-german',
                   framework='pt', device=args.device)

    total = 0
    for path in sorted(args.input_dir.glob('*.wav')):
        if args.skip_a and '_a' in path.stem:
            continue
        waveform, sample_rate = torchaudio.load(path)
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)[0].numpy()
        with torch.no_grad():
            result = asr(audio_16k, return_timestamps='word')
        count = 0
        for chunk in result['chunks']:
            word = chunk['text']
            if 'undzwanzig' not in word:
                continue
            start, end = chunk['timestamp']
            number = process.extractOne(word, TARGET_WORDS)[0]
            torchaudio.save(os.fspath(args.output_dir / f'{path.stem}__{count:03d}_{number}.wav'),
                            one_second(waveform, sample_rate, start, end), sample_rate)
            count += 1
        print(f'{path.stem}: {count} words', flush=True)
        total += count
    print(f'{total} words extracted')


if __name__ == '__main__':
    main()
