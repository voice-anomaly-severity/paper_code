#!/usr/bin/env python3
"""Recording-level score/acoustic associations and rank-adjusted RBH sensitivity analyses."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

import os

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get('VOICE_DATA_ROOT') or Path(__file__).resolve().parents[1])   # data root (default: repository root)
INPUT = HERE/'outputs/available_clips'
SAVED = DATA_ROOT/'img/ukl_wavs/paper/student_learner/simple/tree_dark_bright/correlation_plots_number'
ACOUSTICS = ['f0_median_hz','cpps_db','spectral_slope_db_per_octave','rms_dbfs']
EXPLORATORY = ['f0_iqr_hz','voiced_fraction']
LABELS = {'f0_median_hz':'F0', 'cpps_db':'CPPS', 'spectral_slope_db_per_octave':'Spectral slope',
          'rms_dbfs':'Relative intensity', 'f0_iqr_hz':'F0 variability', 'voiced_fraction':'Voiced fraction'}
PITCH_FLAGS = {'pitch_near_search_bounds','pitch_large_adjacent_jump','pitch_wide_range',
               'no_voiced_frames','low_voiced_fraction'}


def file_hash(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def zrank(values):
    r = stats.rankdata(np.asarray(values,dtype=float), axis=0, method='average')
    sd = r.std(axis=0,ddof=1)
    if np.any(sd == 0):
        raise ValueError('Constant variable in ranked model')
    return (r-r.mean(axis=0))/sd


def pearson(a,b):
    a = a-a.mean(); b = b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/den) if den else np.nan


def rank_model(values, robust=True):
    """Columns: score, outcome, covariates. Rank all columns BEFORE residualization."""
    z = zrank(values)
    x, y = z[:,0], z[:,1]
    cov = np.column_stack([np.ones(len(z)),z[:,2:]])
    xr = x-cov@np.linalg.lstsq(cov,x,rcond=None)[0]
    yr = y-cov@np.linalg.lstsq(cov,y,rcond=None)[0]
    partial = pearson(xr,yr)
    if not robust:
        return partial
    design = np.column_stack([cov,x])
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError('Rank-deficient design')
    pinv = np.linalg.pinv(design)
    beta = pinv@y
    resid = y-design@beta
    hat = np.sum(design*pinv.T,axis=1)
    adjusted = resid/np.maximum(1-hat,1e-10)
    cov_hc3 = (pinv*adjusted)@(pinv*adjusted).T
    se = float(np.sqrt(cov_hc3[-1,-1]))
    p = float(2*stats.norm.sf(abs(beta[-1]/se))) if se else np.nan
    sse = float(resid@resid)
    base_sse = float(yr@yr)
    total = float(((y-y.mean())**2).sum())
    return {'rho':partial,'score_rank_beta':float(beta[-1]),'score_rank_beta_hc3_se':se,
            'p_hc3_normal':p,'r_squared_rank_full':1-sse/total,
            'r_squared_rank_baseline':1-base_sse/total,
            'delta_r_squared_rank_in_sample':(base_sse-sse)/total,
            'max_leverage':float(hat.max()),'design_condition_number':float(np.linalg.cond(design)),
            'n':len(y),'n_covariates':z.shape[1]-2}


def bootstrap_rho(values,n_boot,rng,partial=False):
    values=np.asarray(values,dtype=float)
    result=[]
    for _ in range(n_boot):
        b=values[rng.integers(0,len(values),len(values))]
        try:
            if partial:
                r=rank_model(b,robust=False)
            else:
                z=zrank(b)
                r=pearson(z[:,0],z[:,1])
            if np.isfinite(r): result.append(r)
        except ValueError:
            pass
    if len(result)<n_boot*.95:
        raise ValueError('Too many invalid bootstrap samples')
    lo,hi=np.quantile(result,[.025,.975])
    return float(lo),float(hi),len(result)


def permutation_p(values,n_perm,rng):
    """Two-sided absolute-rho test with add-one correction; no clip pseudoreplication."""
    z=zrank(values)
    x,y=z[:,0],z[:,1]
    observed=abs(pearson(x,y))
    count=0
    for start in range(0,n_perm,500):
        n=min(500,n_perm-start)
        permutations=np.array([rng.permutation(len(y)) for _ in range(n)])
        vals=(y[permutations]@x)/(len(y)-1)
        count+=int((np.abs(vals)>=observed-1e-12).sum())
    return (count+1)/(n_perm+1)


def adjust_families(frame,pcol,groups,outcol):
    frame[outcol]=np.nan
    for _,g in frame.groupby(groups):
        good=g[pcol].notna()
        frame.loc[g.index[good],outcol]=stats.false_discovery_control(g.loc[good,pcol].to_numpy(),method='bh')
    return frame


def load_data():
    clips=pd.read_csv(INPUT/'clip_features.csv')
    clips=clips[(clips.audio_variant=='primary') & clips.score_match_by_filename].copy()
    if not clips.clip_id.is_unique:
        raise ValueError('Repeated clip IDs')
    clinical=pd.read_csv(SAVED/'patient_scores.csv').rename(columns={'patient_id':'source_recording_id'}).set_index('source_recording_id')
    expected=pd.read_csv(SAVED/'recording_scores.csv').groupby('patient_id').size()
    return clips,clinical,expected


def aggregate_common(clips,clinical,expected,cols,median=False):
    good=clips.dropna(subset=cols+['mahalanobis_score'])
    agg=good.groupby('source_recording_id')[cols].agg('median' if median else 'mean')
    agg['score']=good.groupby('source_recording_id').mahalanobis_score.mean()
    agg['n_clips']=good.groupby('source_recording_id').size()
    agg['n_original_clips']=expected.reindex(agg.index)
    agg['coverage']=agg.n_clips/agg.n_original_clips
    return agg.join(clinical[['R','B','H','age_years','sex','condition']],validate='one_to_one').reset_index()


def build_cohorts(clips,clinical,expected):
    primary=aggregate_common(clips,clinical,expected,ACOUSTICS)
    safe=clips[~clips.qc_flags.fillna('').apply(lambda s:bool(set(s.split(';')) & PITCH_FLAGS))]
    technical=clips[~clips.qc_flags.fillna('').str.contains('long_original_clip|samples_near_full_scale')]
    cohorts={
        'primary_common_clips':(primary,ACOUSTICS),
        'complete_clip_coverage':(primary[primary.coverage==1].copy(),ACOUSTICS),
        'pitch_qc_restricted':(aggregate_common(safe,clinical,expected,ACOUSTICS),ACOUSTICS),
        'median_features':(aggregate_common(clips,clinical,expected,ACOUSTICS,median=True),ACOUSTICS),
        'omit_f0':(aggregate_common(clips,clinical,expected,ACOUSTICS[1:]),ACOUSTICS[1:]),
        'technical_qc_restricted':(aggregate_common(technical,clinical,expected,ACOUSTICS),ACOUSTICS),
    }
    return cohorts


def feature_associations(clips,clinical,expected,n_boot,n_perm,rng,out):
    rows=[]
    for feat in ACOUSTICS+EXPLORATORY:
        frame=aggregate_common(clips,clinical,expected,[feat])
        frame.to_csv(out/f'score_feature_dataset_{feat}.csv',index=False)
        vals=frame[['score',feat]].to_numpy()
        r=float(stats.spearmanr(vals[:,0],vals[:,1]).statistic)
        lo,hi,b=bootstrap_rho(vals,n_boot,rng)
        p=permutation_p(vals,n_perm,rng)
        rows.append({'feature':feat,'label':LABELS[feat],'family':'primary_four' if feat in ACOUSTICS else 'exploratory_two',
                     'n_recordings':len(frame),'n_clips':int(frame.n_clips.sum()),'rho':r,
                     'ci95_low':lo,'ci95_high':hi,'p_permutation':p,'bootstrap_valid':b})
    return adjust_families(pd.DataFrame(rows),'p_permutation',['family'],'q_bh')


def rbh_models(cohorts,n_boot,rng,out):
    rows=[];flow=[];diagnostics=[]
    for name,(frame,acoustic_cols) in cohorts.items():
        # Use identical rows for ALL adjustment specifications within each outcome/scenario.
        frame.to_csv(out/f'analysis_dataset_{name}.csv',index=False)
        specs={'unadjusted':[], 'acoustic':acoustic_cols,
               'age_sex':['age_years','sex'], 'acoustic_age_sex':acoustic_cols+['age_years','sex']}
        for outcome in ['R','B','H']:
            complete=frame.dropna(subset=['score',outcome]+acoustic_cols+['age_years','sex'])
            flow.append({'scenario':name,'outcome':outcome,'n_recordings_available':len(frame),
                         'n_recordings_analyzed':len(complete),'n_missing_outcome_or_covariate':len(frame)-len(complete),
                         'n_clips_analyzed':int(complete.n_clips.sum()),
                         'n_complete_clip_coverage':int((complete.coverage==1).sum()),
                         'median_clips_per_recording':float(complete.n_clips.median())})
            for model,covars in specs.items():
                vals=complete[['score',outcome]+covars].to_numpy(float)
                res=rank_model(vals)
                lo,hi,n_valid=bootstrap_rho(vals,n_boot,rng,partial=True)
                rows.append({'scenario':name,'outcome':outcome,'model':model,'covariates':';'.join(covars),
                             **res,'ci95_low':lo,'ci95_high':hi,'bootstrap_valid':n_valid})
            print(f'{name}: {outcome}, n={len(complete)}',flush=True)
        clean=frame.dropna(subset=['score']+acoustic_cols+['age_years','sex'])
        predcols=['score']+acoustic_cols+['age_years','sex']
        z=zrank(clean[predcols].to_numpy())
        for j,col in enumerate(predcols):
            others=np.column_stack([np.ones(len(z)),np.delete(z,j,axis=1)])
            resid=z[:,j]-others@np.linalg.lstsq(others,z[:,j],rcond=None)[0]
            vif=float((z[:,j]@z[:,j])/(resid@resid))
            diagnostics.append({'scenario':name,'predictor':col,'n':len(clean),'rank_predictor_vif':vif})
    result=adjust_families(pd.DataFrame(rows),'p_hc3_normal',['scenario','model'],'q_bh_across_R_B_H')
    pd.DataFrame(flow).to_csv(out/'analysis_counts.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(out/'collinearity_diagnostics.csv',index=False)
    return result


def missingness_tables(clips,clinical,cohorts,out):
    before=clips.groupby('source_recording_id').agg(n_clips=('clip_id','size'),n_no_f0=('f0_median_hz',lambda s:s.isna().sum()))
    before['n_pitch_flags']=clips.assign(flag=clips.qc_flags.fillna('').apply(lambda s:bool(set(s.split(';'))&PITCH_FLAGS))).groupby('source_recording_id').flag.sum()
    before['pitch_flag_fraction']=before.n_pitch_flags/before.n_clips
    before=before.join(clinical[['R','B','H','condition']])
    before['retained_in_pitch_qc_analysis']=before.index.isin(cohorts['pitch_qc_restricted'][0].source_recording_id)
    before.to_csv(out/'recording_quality_and_missingness.csv')
    before.groupby('H',dropna=False).agg(n_recordings=('n_clips','size'),n_clips=('n_clips','sum'),
             n_pitch_flags=('n_pitch_flags','sum'),n_no_f0=('n_no_f0','sum'),
             mean_recording_pitch_flag_fraction=('pitch_flag_fraction','mean'),
             n_recordings_retained_after_pitch_qc=('retained_in_pitch_qc_analysis','sum')).to_csv(out/'quality_by_H.csv')


def plots(cohorts,pairs,models,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame=cohorts['primary_common_clips'][0]
    fig,axs=plt.subplots(2,2,figsize=(10,8))
    for ax,col in zip(axs.flat,ACOUSTICS):
        pair_frame=pd.read_csv(out/f'score_feature_dataset_{col}.csv')
        ax.scatter(pair_frame[col],pair_frame.score,s=16,alpha=.55,color='#286181')
        r=pairs[pairs.feature==col].iloc[0]
        ax.set_xlabel(LABELS[col]+({'f0_median_hz':' (Hz)','cpps_db':' (dB)','spectral_slope_db_per_octave':' (dB/octave)','rms_dbfs':' (dBFS)'}[col]))
        ax.set_ylabel('Mean anomaly score')
        ax.set_title(f"Spearman ρ={r.rho:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}], n={int(r.n_recordings)}",fontsize=10)
    fig.suptitle('Available pathological recordings — provisional filename-linked analysis')
    fig.tight_layout();fig.savefig(out/'score_feature_correlations.png',dpi=180);fig.savefig(out/'score_feature_correlations.pdf');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
    labels=['Unadjusted','Acoustic adjusted','Age/sex adjusted','Acoustic + age/sex']
    order=['unadjusted','acoustic','age_sex','acoustic_age_sex']
    for ax,outcome in zip(axs,['R','B','H']):
        subset=models[(models.scenario=='primary_common_clips')&(models.outcome==outcome)].set_index('model').loc[order]
        yy=np.arange(len(subset))
        ax.errorbar(subset.rho,yy,xerr=np.vstack([subset.rho-subset.ci95_low,subset.ci95_high-subset.rho]),fmt='o',capsize=4,color='#286181')
        ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(yy,labels);ax.invert_yaxis()
        ax.set_xlim(-.3,.7);ax.set_title(f'{outcome}: n={int(subset.iloc[0]["n"])}');ax.set_xlabel('Rank correlation / partial rank correlation')
    fig.suptitle('Score–RBH associations: identical recordings within each comparison')
    fig.tight_layout();fig.savefig(out/'rbh_adjustment.png',dpi=180);fig.savefig(out/'rbh_adjustment.pdf');plt.close(fig)
    matrix=frame[['score']+ACOUSTICS+['age_years','sex']].corr(method='spearman')
    matrix.to_csv(out/'predictor_spearman_matrix.csv')
    fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(matrix,vmin=-1,vmax=1,cmap='RdBu_r')
    names=['Score']+[LABELS[k] for k in ACOUSTICS]+['Age','Sex']
    ax.set_xticks(range(len(names)),names,rotation=45,ha='right');ax.set_yticks(range(len(names)),names)
    for i in range(len(names)):
        for j in range(len(names)):ax.text(j,i,f'{matrix.iloc[i,j]:.2f}',ha='center',va='center',fontsize=9)
    fig.colorbar(im,ax=ax,label='Spearman correlation');fig.tight_layout();fig.savefig(out/'predictor_correlations.png',dpi=160);plt.close(fig)


def write_report(pairs,models,out,n_boot,n_perm):
    primary=models[models.scenario=='primary_common_clips']
    lines=['# Acoustic association results', '',
           '**Provisional analysis of available pathological recordings. Audio-to-score links are filename-based; exact original scored waveform versions remain unverified.**', '',
           'Analysis unit: source recording, following the user’s clarification that recordings represent different people. Word clips are averaged within each source and are never treated as independent participants. If that identity assignment changes, inference must be rerun at the corrected participant level.', '',
           '## Score–feature associations', '',
           '| Feature | Recordings | Spearman rho | 95% bootstrap CI | Permutation p | BH q |',
           '| --- | ---: | ---: | --- | ---: | ---: |']
    for r in pairs.itertuples():
        lines.append(f'| {r.label} | {r.n_recordings} | {r.rho:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.p_permutation:.5g} | {r.q_bh:.5g} |')
    lines += ['', 'The four requested features form one multiple-testing family. F0 variability and voiced fraction are exploratory and form a separate two-test family. Each pair uses features and mean scores computed from the same valid clips. The source-level mean of clip median F0 is used as the primary F0 summary.', '',
              '## Score–RBH associations on identical complete cases', '',
              '| Rating | Model | Recordings | Rank/partial rank correlation | 95% bootstrap CI | HC3 p | BH q across R/B/H |',
              '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    for r in primary.itertuples():
        lines.append(f'| {r.outcome} | {r.model} | {r.n} | {r.rho:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] | {r.p_hc3_normal:.5g} | {r.q_bh_across_R_B_H:.5g} |')
    lines += ['', 'Acoustic adjustment includes F0, CPPS, spectral slope and relative RMS intensity. Age and sex are secondary adjustment variables; this is not an age-stratified analysis. All four models for a given scenario/rating use exactly the same recording rows. No healthy controls are added to the adjusted analysis.', '',
              '## Sensitivity analyses: acoustic-adjusted association', '',
              '| Scenario | Rating | Recordings | Partial rank correlation | 95% bootstrap CI |',
              '| --- | --- | ---: | ---: | --- |']
    for r in models[(models.model=='acoustic')&(models.scenario!='primary_common_clips')].itertuples():
        lines.append(f'| {r.scenario} | {r.outcome} | {r.n} | {r.rho:.3f} | [{r.ci95_low:.3f}, {r.ci95_high:.3f}] |')
    lines += ['', '## Statistical specification', '',
      f'- Two-sided unadjusted association tests use {n_perm:,} random permutations with an add-one correction. A p-value at the minimum resolution is not zero.',
      f'- Confidence intervals are percentile intervals from {n_boot:,} whole-recording bootstrap resamples. Rankings and adjustment models are recomputed in each resample; ties receive average ranks.',
      '- Adjusted associations are Pearson correlations of residuals after ranking the score, rating and covariates and regressing each ranked score/rating on ranked covariates plus an intercept. These are partial rank correlations. This differs from residualizing raw values and then ranking residuals.',
      '- Adjusted-model p-values use the score coefficient in the corresponding rank-linear regression with HC3 heteroscedasticity-robust covariance and an asymptotic normal reference distribution. They test a model-conditional zero rank association and are not permutation p-values. The same test is reported for the unadjusted rank model for consistency.',
      '- BH correction for RBH outcomes is applied within each model/scenario across R, B and H. These families do not constitute a single study-wide multiplicity adjustment; sensitivity results are descriptive robustness checks, not alternative routes to a favorable finding.',
      '- Primary multivariable aggregates use only clips complete for all four acoustic features, and average scores over those same clips. Acoustic features receive equal clip weights; source recordings receive equal analysis weights. No imputation or outcome-driven covariate selection is used.',
      '- Sensitivities: complete original clip coverage; remove clips flagged for pitch bounds, jumps, wide range, little/no voicing; median feature aggregation (mean score retained); omit F0 and retain clips without pitch; remove unusually long or near-full-scale clips. Each filtered analysis recomputes both features and scores on its retained clips.',
      '- QC filtering may preferentially remove difficult/pathological phonation. It is a sensitivity analysis, not a declaration that flagged samples are invalid. `quality_by_H.csv` documents this selection.',
      '- Collinearity diagnostics are VIFs for ranked predictors. Saved rank-model R-squared and changes in R-squared are in-sample descriptions, not cross-validated predictive value.', '',
      '## Limits on interpretation', '',
      '- Exact waveform provenance is unverified. The measurements describe the hashed recovered audio; correlations with old model scores remain provisional until audio versions are confirmed or those recordings are rescored.',
      '- Relative RMS is a digital amplitude measure, not calibrated SPL or the clinical maximum intensity measure. Previously applied recording gain/normalization may affect its interpretation. Task/intensity labels are unavailable.',
      '- Acoustic adjustment can remove shared manifestations of dysphonia itself. Attenuation does not prove the model is invalid; a surviving association does not establish causality, independence from every confounder, or incremental out-of-sample predictive value.',
      '- These analyses characterize the available pathological subset. They do not validate healthy-versus-pathological discrimination, an older healthy reference, or preclinical detection.',
      '- No inference is made from thousands of clips as if they were independent people. Confidence intervals assume independent source-recording units as clarified by the user.', '',
      'Methods references: [SciPy Spearman/permutation guidance](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), [BH adjustment](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.false_discovery_control.html), [HC3 covariance](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hc3.html).', '',
      'Reproduce from the project root:', '', '```bash',
      'OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/voice-acoustics-mpl \\',
      '  /tmp/voice-acoustics-venv/bin/python acoustic_analysis/analyze_associations.py', '```',
      '', 'If the temporary environment is gone, recreate the Python 3.12 environment from `acoustic_analysis/requirements.txt` and substitute its interpreter.']
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,default=HERE/'outputs/associations')
    ap.add_argument('--bootstrap',type=int,default=2000)
    ap.add_argument('--permutations',type=int,default=19999)
    ap.add_argument('--seed',type=int,default=20261006)
    args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
    start=time.monotonic();rng=np.random.default_rng(args.seed)
    config={'created_utc':datetime.now(timezone.utc).isoformat(),'bootstrap':args.bootstrap,
            'permutations':args.permutations,'seed':args.seed,'script_sha256':file_hash(__file__),
            'clip_features_sha256':file_hash(INPUT/'clip_features.csv'),
            'clinical_scores_sha256':file_hash(SAVED/'patient_scores.csv'),
            'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','scipy','pandas','matplotlib']},
            'unit':'source recording; assumed independent per user clarification',
            'scope':'pathological primary clips; filename-linked exploratory analysis',
            'primary_features':ACOUSTICS,'exploratory_features':EXPLORATORY,
            'scenarios':['primary_common_clips','complete_clip_coverage','pitch_qc_restricted','median_features','omit_f0','technical_qc_restricted'],
            'outcomes':['R','B','H'],'adjustments':['none','acoustic','age_sex','acoustic_age_sex'],
            'inference':'whole-recording percentile bootstrap; permutations for score-feature pairs; HC3 normal p for rank regressions',
            'bh_families':'four primary feature pairs; two exploratory pairs; three RBH outcomes per model per scenario',
            'pitch_exclusion_flags':sorted(PITCH_FLAGS),'command':sys.argv}
    (out/'analysis_config.json').write_text(json.dumps(config,indent=2)+'\n')
    clips,clinical,expected=load_data();cohorts=build_cohorts(clips,clinical,expected)
    missingness_tables(clips,clinical,cohorts,out)
    print('Calculating score-feature correlations',flush=True)
    pairs=feature_associations(clips,clinical,expected,args.bootstrap,args.permutations,rng,out)
    pairs.to_csv(out/'score_feature_correlations.csv',index=False)
    print('Calculating RBH models and sensitivities',flush=True)
    models=rbh_models(cohorts,args.bootstrap,rng,out)
    models.to_csv(out/'rbh_adjusted_associations.csv',index=False)
    plots(cohorts,pairs,models,out)
    write_report(pairs,models,out,args.bootstrap,args.permutations)
    summary={'n_primary_scored_clips':len(clips),'n_primary_source_recordings':clips.source_recording_id.nunique(),
             'n_pairs':len(pairs),'n_rbh_models':len(models),'runtime_seconds':time.monotonic()-start,
             'primary_pairs':pairs[pairs.family=='primary_four'].to_dict(orient='records'),
             'primary_rbh':models[models.scenario=='primary_common_clips'].to_dict(orient='records')}
    (out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(f'Finished in {time.monotonic()-start:.1f} seconds: {out}',flush=True)


if __name__=='__main__':
    main()
