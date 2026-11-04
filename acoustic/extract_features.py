#!/usr/bin/env python3
"""Extract auditable acoustic features; never alter input recordings or model scores."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import pandas as pd
import parselmouth
from parselmouth.praat import call
from scipy import signal
import soundfile as sf

ROOT = Path(os.environ.get('VOICE_DATA_ROOT') or Path(__file__).resolve().parents[1])   # data root (default: repository root)
RESULTS = Path('img/ukl_wavs/paper/student_learner/simple/tree_dark_bright/correlation_plots_number')
PRIMARY = Path('audio_img/number_wav/unhealthy/number')
SUPPLEMENTAL = Path('audio_img/number_wav_noise_reduce/unhealthy/number')
FEATURES = ['f0_median_hz', 'f0_iqr_hz', 'f0_p05_hz', 'f0_p95_hz',
            'voiced_fraction', 'cpps_db', 'spectral_slope_db_per_octave', 'rms_dbfs']
SETTINGS = {
    'window': 'native-sample-rate mono centered crop of at most 1 second; no added padding',
    'target_seconds': 1.0,
    'boundary_trim': 'only exactly zero-valued leading/trailing runs >=10 ms, after cropping',
    'dc_removal': 'subtract mean of analyzed interval for every feature',
    'pitch': {'method': 'Praat raw autocorrelation', 'step_seconds': .01,
              'floor_hz': 60., 'ceiling_hz': 600., 'max_candidates': 15,
              'very_accurate': False, 'silence_threshold': .03, 'voicing_threshold': .45,
              'octave_cost': .01, 'octave_jump_cost': .35, 'voiced_unvoiced_cost': .14},
    'cpps': {'pitch_floor_hz': 60., 'step_seconds': .002, 'max_frequency_hz': 5000.,
             'preemphasis_from_hz': 50., 'subtract_trend_before_smoothing': True,
             'time_smoothing_seconds': .01, 'quefrency_smoothing_seconds': .001,
             'peak_pitch_floor_hz': 60., 'peak_pitch_ceiling_hz': 600.,
             'tolerance': .05, 'interpolation': 'Parabolic',
             'trend_quefrency_start_seconds': .001, 'trend_quefrency_end_seconds': 0.,
             'trend_type': 'Straight', 'fit_method': 'Robust'},
    'spectral_slope': {'method': 'OLS of Welch PSD dB on log2(frequency/1000)',
                       'frequency_band_hz': [500., 4000.], 'window': 'hann',
                       'frame_seconds': .04, 'hop_seconds': .01,
                       'nfft': 2048, 'preemphasis': False,
                       'frames': 'all frames in analyzed interval, not only voiced frames'},
    'intensity': '20*log10(RMS), digital amplitude reference 1; not calibrated SPL',
    'minimum_duration_seconds': .1,
    'qc_flags_only': {'low_voiced_fraction': .2, 'pitch_near_bounds_fraction': .1,
                      'adjacent_voiced_pitch_jump_octaves': .5, 'pitch_p95_to_p05_ratio': 2.,
                      'clipping_level': .999, 'long_original_clip_seconds': 2.},
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_signal(samples, sr):
    """Return actual samples and their interval in the input clip (never zero-pad)."""
    x = np.asarray(samples, dtype=float)
    if x.ndim == 2:
        x = x.mean(axis=1)
    n = min(len(x), round(sr * SETTINGS['target_seconds']))
    start = (len(x) - n) // 2
    end = start + n
    x = x[start:end]
    nz = np.flatnonzero(x != 0)
    left = right = 0
    if len(nz):
        left = int(nz[0]) if nz[0] >= sr * .01 else 0
        trailing = len(x) - int(nz[-1]) - 1
        right = trailing if trailing >= sr * .01 else 0
        end -= right
        start += left
        x = x[left:len(x)-right if right else None]
    return x, {'analysis_start_seconds': start/sr, 'analysis_end_seconds': end/sr,
               'analysis_duration_seconds': len(x)/sr,
               'exact_zero_boundary_trim_seconds': (left+right)/sr}


def acoustic_features(samples, sr, return_track=False):
    x, info = prepare_signal(samples, sr)
    result = dict.fromkeys(FEATURES, np.nan)
    result.update(info)
    flags = []
    errors = []
    track = None
    result['clipping_fraction'] = float(np.mean(np.abs(x) >= .999)) if len(x) else np.nan
    result['peak_amplitude'] = float(np.max(np.abs(x))) if len(x) else np.nan
    if len(x)/sr < SETTINGS['minimum_duration_seconds']:
        flags.append('too_short')
    if not np.isfinite(x).all():
        flags.append('nonfinite_audio')
    if flags:
        return result, flags, errors, track
    x = x - x.mean()
    rms = float(np.sqrt(np.mean(x*x)))
    if rms < 1e-12:
        flags.append('silent_or_constant_audio')
        return result, flags, errors, track
    result['rms_dbfs'] = float(20*np.log10(rms))
    snd = parselmouth.Sound(x, sampling_frequency=sr)
    try:
        p = SETTINGS['pitch']
        pitch = snd.to_pitch_ac(time_step=p['step_seconds'], pitch_floor=p['floor_hz'],
                                max_number_of_candidates=p['max_candidates'],
                                very_accurate=p['very_accurate'], silence_threshold=p['silence_threshold'],
                                voicing_threshold=p['voicing_threshold'], octave_cost=p['octave_cost'],
                                octave_jump_cost=p['octave_jump_cost'],
                                voiced_unvoiced_cost=p['voiced_unvoiced_cost'], pitch_ceiling=p['ceiling_hz'])
        f = pitch.selected_array['frequency']
        voiced = f[f > 0]
        result['pitch_frames'] = len(f)
        result['voiced_frames'] = len(voiced)
        result['voiced_fraction'] = len(voiced)/len(f) if len(f) else np.nan
        if len(voiced):
            q = np.quantile(voiced, [.05, .25, .5, .75, .95])
            result.update(f0_p05_hz=q[0], f0_median_hz=q[2], f0_iqr_hz=q[3]-q[1], f0_p95_hz=q[4])
            result['pitch_near_bounds_fraction'] = float(np.mean((voiced < p['floor_hz']*1.05) | (voiced > p['ceiling_hz']*.95)))
            if result['pitch_near_bounds_fraction'] > .1:
                flags.append('pitch_near_search_bounds')
            adjacent = (f[:-1] > 0) & (f[1:] > 0)
            jumps = np.abs(np.log2(f[1:][adjacent]/f[:-1][adjacent]))
            result['pitch_large_jump_count'] = int(np.sum(jumps > .5))
            if result['pitch_large_jump_count']:
                flags.append('pitch_large_adjacent_jump')
            if q[4]/q[0] > 2:
                flags.append('pitch_wide_range')
        else:
            flags.append('no_voiced_frames')
        if result['voiced_fraction'] < .2:
            flags.append('low_voiced_fraction')
        if return_track:
            track = (pitch.xs() + info['analysis_start_seconds'], np.where(f > 0, f, np.nan))
    except Exception as e:
        errors.append('pitch: '+str(e))
    try:
        c = SETTINGS['cpps']
        pc = call(snd, 'To PowerCepstrogram', c['pitch_floor_hz'], c['step_seconds'],
                  c['max_frequency_hz'], c['preemphasis_from_hz'])
        result['cpps_db'] = float(call(pc, 'Get CPPS', 'yes', c['time_smoothing_seconds'],
                    c['quefrency_smoothing_seconds'], c['peak_pitch_floor_hz'], c['peak_pitch_ceiling_hz'],
                    c['tolerance'], c['interpolation'], c['trend_quefrency_start_seconds'],
                    c['trend_quefrency_end_seconds'], c['trend_type'], c['fit_method']))
    except Exception as e:
        errors.append('cpps: '+str(e))
    try:
        nperseg = min(len(x), round(sr*.04))
        hop = min(nperseg, max(1, round(sr*.01)))
        freq, psd = signal.welch(x, fs=sr, window='hann', nperseg=nperseg,
                                noverlap=nperseg-hop, nfft=max(2048, nperseg),
                                detrend=False, scaling='density', average='mean')
        mask = (freq >= 500) & (freq <= 4000) & (psd > 0)
        if sr/2 < 4000 or mask.sum() < 3:
            raise ValueError('insufficient spectral bandwidth')
        a = np.log2(freq[mask]/1000)
        b = 10*np.log10(psd[mask])
        slope, intercept = np.polyfit(a, b, 1)
        pred = slope*a+intercept
        result['spectral_slope_db_per_octave'] = float(slope)
        result['spectral_slope_r_squared'] = float(1-((b-pred)**2).sum()/((b-b.mean())**2).sum())
    except Exception as e:
        errors.append('spectral_slope: '+str(e))
    if result['clipping_fraction'] > 0:
        flags.append('samples_near_full_scale')
    if info['exact_zero_boundary_trim_seconds'] > 0:
        flags.append('exact_zero_boundary_trimmed')
    return result, flags, errors, track


def process_item(item):
    row = dict(item)
    try:
        p = Path(row['audio_path'])
        samples, sr = sf.read(p, dtype='float64', always_2d=True)
        row.update(audio_sha256=sha256(p), sample_rate_hz=sr, channels=samples.shape[1],
                   original_duration_seconds=len(samples)/sr, audio_subtype=sf.info(p).subtype)
        f, flags, errors, _ = acoustic_features(samples, sr)
        if len(samples)/sr > 2:
            flags.append('long_original_clip')
        row.update(f, qc_flags=';'.join(flags), extraction_errors=';'.join(errors),
                   extraction_status='partial' if errors or any(not np.isfinite(f[k]) for k in FEATURES) else 'ok')
    except Exception as e:
        row.update(extraction_status='error', extraction_errors=str(e), qc_flags='read_or_extraction_error')
    return row


def make_manifest(root):
    scores = pd.read_csv(root/RESULTS/'recording_scores.csv').set_index('recording_id')
    if not scores.index.is_unique:
        raise ValueError('Duplicate saved score IDs')
    clinical = pd.read_csv(root/RESULTS/'patient_scores.csv').set_index('patient_id')
    if not clinical.index.is_unique:
        raise ValueError('Duplicate source recording metadata IDs')
    items, used = [], set()
    for variant, folder in [('primary', PRIMARY), ('supplemental_noise_reduced', SUPPLEMENTAL)]:
        for p in sorted((root/folder).rglob('*.wav')):
            clip = p.stem
            if variant != 'primary' and (clip not in scores.index or clip in used):
                continue
            if clip in used:
                raise ValueError('Duplicate clip ID within primary folder: '+clip)
            used.add(clip)
            source, word, token = clip.rsplit('_', 2)
            matched = clip in scores.index
            if matched and scores.loc[clip, 'patient_id'] != source:
                raise ValueError('Filename/source mapping disagrees with saved table: '+clip)
            row = {'clip_id': clip, 'source_recording_id': source, 'word': word,
                   'extraction_index': token, 'audio_variant': variant,
                   'audio_path': str(p.resolve()), 'score_match_by_filename': matched,
                   'exact_scored_waveform_verified': False,
                   'mahalanobis_score': float(scores.loc[clip, 'mahalanobis_score']) if matched else np.nan}
            if source in clinical.index:
                for col in ['R','B','H','RBH','condition']:
                    row[col] = clinical.loc[source, col]
            items.append(row)
    if not items:
        raise ValueError('No candidate audio files found')
    missing = scores.loc[~scores.index.isin(used)].reset_index()
    return items, scores, clinical, missing


def pilot_selection(items, limit):
    """Prespecified H-stratified sampling for QC only, not tuning association results."""
    rng = random.Random(42)
    buckets = {}
    for item in items:
        key = str(item.get('H', 'missing'))
        buckets.setdefault(key, []).append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    while len(selected) < min(limit, len(items)):
        for key in sorted(buckets):
            if buckets[key] and len(selected) < limit:
                selected.append(buckets[key].pop())
    return selected


def aggregate(features, clinical, scores, out):
    # Supplemental processing variants are intentionally excluded from the primary table.
    primary = features[(features.audio_variant == 'primary') & features.score_match_by_filename & (features.extraction_status != 'error')]
    rows = []
    expected = scores.groupby('patient_id').size()
    for source, group in primary.groupby('source_recording_id'):
        row = {'source_recording_id': source, 'n_feature_clips': len(group),
               'n_original_scored_clips': int(expected.loc[source]),
               'clip_coverage_fraction': len(group)/expected.loc[source],
               'n_qc_flagged_clips': int((group.qc_flags != '').sum()),
               'mean_score_matched_clips': group.mahalanobis_score.mean()}
        for col in FEATURES:
            row['mean_'+col] = group[col].mean()
            row['median_'+col] = group[col].median()
            valid = group[col].notna()
            row['n_'+col] = int(valid.sum())
            row['mean_score_valid_'+col] = group.loc[valid, 'mahalanobis_score'].mean()
        for col in ['R','B','H','RBH','condition','mean_mahalanobis_score']:
            row['original_'+col if col == 'mean_mahalanobis_score' else col] = clinical.loc[source, col]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out/'recording_features_primary.csv', index=False)


def create_plots(features, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary = features[features.audio_variant == 'primary']
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, col in zip(axes.flat, ['f0_median_hz','f0_iqr_hz','cpps_db','spectral_slope_db_per_octave','rms_dbfs','voiced_fraction']):
        values = primary[col].dropna()
        ax.hist(values, bins=40, color='#366b87')
        ax.set_title(col.replace('_',' '), fontsize=10)
        ax.set_ylabel('Clips')
    fig.suptitle('Acoustic feature distributions — primary available clips')
    fig.tight_layout()
    fig.savefig(out/'feature_distributions.png', dpi=160)
    plt.close(fig)
    # Deterministic exemplars across F0, including low voicing and edge cases.
    candidates = primary.dropna(subset=['f0_median_hz']).sort_values('f0_median_hz')
    picks = candidates.iloc[np.linspace(0, max(0,len(candidates)-1), min(9,len(candidates))).astype(int)].copy()
    picks = pd.concat([picks,primary.sort_values('voiced_fraction').head(3)]).drop_duplicates('clip_id').head(12)
    fig, axes = plt.subplots(len(picks), 1, figsize=(12, max(3,len(picks)*1.8)), squeeze=False)
    for ax, (_, row) in zip(axes[:,0], picks.iterrows()):
        x, sr = sf.read(row.audio_path, always_2d=True)
        x = x.mean(axis=1)
        y, interval = prepare_signal(x,sr)
        t = np.arange(len(y))/sr+interval['analysis_start_seconds']
        ax.plot(t,y,color='#678896',linewidth=.4)
        _, _, _, track = acoustic_features(x,sr,return_track=True)
        twin = ax.twinx()
        if track is not None:
            twin.plot(track[0],track[1],'.',color='#b23a37',markersize=2)
        twin.set_ylim(50,650)
        twin.set_ylabel('F0 Hz',fontsize=8)
        ax.set_title(row.clip_id+' | '+str(row.qc_flags),fontsize=8,loc='left')
        ax.set_xlim(interval['analysis_start_seconds'],interval['analysis_end_seconds'])
    fig.tight_layout()
    fig.savefig(out/'pitch_qc_examples.png',dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,default=ROOT)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--limit',type=int,default=0,help='Nonzero selects a reproducible H-stratified QC pilot')
    args = ap.parse_args()
    start = time.monotonic()
    out = args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    items, scores, clinical, missing = make_manifest(args.root)
    if args.limit:
        items = pilot_selection(items,args.limit)
    pd.DataFrame(items).to_csv(out/'input_manifest.csv',index=False)
    missing.to_csv(out/'missing_scored_clips.csv',index=False)
    packages = {p:importlib.metadata.version(p) for p in ['praat-parselmouth','numpy','scipy','pandas','soundfile','matplotlib']}
    config = {'created_utc':datetime.now(timezone.utc).isoformat(), 'settings':SETTINGS,
              'python':platform.python_version(),'packages':packages,'praat_version':parselmouth.PRAAT_VERSION,
              'command':sys.argv,'script_sha256':sha256(__file__),
              'score_table_sha256':sha256(args.root/RESULTS/'recording_scores.csv'),
              'clinical_table_sha256':sha256(args.root/RESULTS/'patient_scores.csv'),
              'scope':'available pathological clips; exact scored waveform provenance unverified',
              'pilot':bool(args.limit),'workers':args.workers}
    (out/'extraction_config.json').write_text(json.dumps(config,indent=2)+'\n')
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,row in enumerate(pool.map(process_item,items,chunksize=8),1):
            rows.append(row)
            if i%100 == 0 or i==len(items):
                print(f'{i}/{len(items)} clips, {time.monotonic()-start:.1f} seconds',flush=True)
    features = pd.DataFrame(rows)
    features.to_csv(out/'clip_features.csv',index=False)
    aggregate(features,clinical,scores,out)
    features[(features.qc_flags != '') | (features.extraction_status != 'ok')].to_csv(out/'qc_flagged_clips.csv',index=False)
    create_plots(features,out)
    summary = {'clip_rows':len(features),'variants':features.audio_variant.value_counts().to_dict(),
               'status_counts':features.extraction_status.value_counts().to_dict(),
               'scored_clip_matches':int(features.score_match_by_filename.sum()),
               'source_recordings':int(features.source_recording_id.nunique()),
               'primary_matched_source_recordings':int(features.loc[(features.audio_variant=='primary') & features.score_match_by_filename,'source_recording_id'].nunique()),
               'flagged_clips':int((features.qc_flags != '').sum()),
               'missing_feature_counts':{k:int(features[k].isna().sum()) for k in FEATURES},
               'feature_quantiles':{k:{str(q):float(v) for q,v in features[k].quantile([0,.05,.5,.95,1]).items()} for k in FEATURES},
               'seconds':time.monotonic()-start,'clinical_validation':'automated extraction and limited waveform/F0 visual QC; no expert phonetic review'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    main()
