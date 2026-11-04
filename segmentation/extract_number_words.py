"""Cut recordings of the counting task into single number words (21-29).

Faster-Whisper (large-v3, German, VAD) transcribes each recording with word timestamps, WhisperX
refines them with a German alignment model, and every word recognised as a number from 20 to 29
is saved as a 1-s clip (centred, zero-padded or trimmed) at the recording's original sampling rate.

    python extract_number_words.py --input-dir <wav folder> --output-dir data/ukl_wavs/healthy_studis/number
    python extract_number_words.py --input-dir <wav folder> --output-dir data/ukl_wavs/unhealthy/number --skip-a
"""
import argparse
from pathlib import Path

import torch
import torchaudio
import whisperx
from faster_whisper import WhisperModel

NUMBER_WORDS = ["zwanzig", "einundzwanzig", "zweiundzwanzig", "dreiundzwanzig", "vierundzwanzig",
                "fünfundzwanzig", "sechsundzwanzig", "siebenundzwanzig", "achtundzwanzig", "neunundzwanzig"]


def number_of(word: str):
    """20..29 if Whisper transcribed the word as such a number, else None."""
    text = word.lower().strip().replace(",", "").replace(".", "")
    if text.isdigit() and 20 <= int(text) <= 29:
        return int(text)
    return None


def one_second_clip(waveform: torch.Tensor, sample_rate: int, start_s: float, end_s: float) -> torch.Tensor:
    """Samples between the word timestamps, centred in exactly 1 s (zero-padded or trimmed)."""
    n = waveform.shape[1]
    segment = waveform[:, max(0, int(round(start_s * sample_rate))):min(n, int(round(end_s * sample_rate)))]
    target = sample_rate
    current = segment.shape[1]
    if current <= 0:
        return torch.zeros((waveform.shape[0], target), dtype=waveform.dtype)
    if current < target:
        before = (target - current) // 2
        return torch.nn.functional.pad(segment, (before, target - current - before))
    start = (current - target) // 2
    return segment[:, start:start + target]


def transcribe_words(audio_16k, whisper, align_model, align_metadata, device):
    """Words with start and end times (s): Faster-Whisper, refined by WhisperX alignment."""
    segments, _ = whisper.transcribe(audio_16k, language="de", vad_filter=True,
                                     vad_parameters=dict(min_silence_duration_ms=400),
                                     word_timestamps=True, beam_size=5, temperature=0.0)
    segments = [{"text": s.text, "start": float(s.start), "end": float(s.end),
                 "words": [{"word": w.word, "start": float(w.start), "end": float(w.end)}
                           for w in (s.words or []) if w.start is not None and w.end is not None]}
                for s in segments]
    aligned = whisperx.align(segments, align_model, align_metadata, audio_16k, device=device,
                             return_char_alignments=False).get("segments", [])
    words = [w for s in aligned for w in s.get("words", []) if w.get("start") is not None]
    if not any(number_of(w.get("word") or "") for w in words):
        # Digits are not in the alignment model's vocabulary; keep the Whisper timestamps then.
        words = [w for s in segments for w in s["words"]]
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-a", action="store_true",
                        help="skip recordings whose name contains '_a' (done for the patient cohort)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    whisper = WhisperModel("large-v3", device=args.device,
                           compute_type="float16" if args.device == "cuda" else "int8")
    align_model, align_metadata = whisperx.load_align_model(language_code="de", device=args.device)

    total = 0
    for path in sorted(args.input_dir.glob("*.wav")):
        if args.skip_a and "_a" in path.stem:
            continue
        waveform, sample_rate = torchaudio.load(path)
        mono = waveform.mean(dim=0, keepdim=True)
        audio_16k = torchaudio.transforms.Resample(sample_rate, 16000)(mono).squeeze(0).float().numpy()

        count = 0
        for word in transcribe_words(audio_16k, whisper, align_model, align_metadata, args.device):
            number = number_of(word["word"])
            if number is None:
                continue
            name = NUMBER_WORDS[number - 20]
            out_dir = args.output_dir / name
            out_dir.mkdir(parents=True, exist_ok=True)
            clip = one_second_clip(waveform, sample_rate, float(word["start"]), float(word["end"]))
            torchaudio.save(out_dir / f"{path.stem}_{name}_{count}.wav", clip, sample_rate)
            count += 1
        print(f"{path.stem}: {count} words", flush=True)
        total += count
    print(f"{total} words extracted")


if __name__ == "__main__":
    main()
