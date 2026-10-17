"""Reproduce the ORIGINAL conditional paired repair-ratio interval; no new inference rule.

Only the training-seed rows are resampled. Evaluation keys, selected setting and horizon are
fixed. The same sampled indices must be used for numerator and denominator. This audit never
rewrites the frozen analysis or changes its decision.
"""
from pathlib import Path
import hashlib
import json
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def paired_ratio(cost, rescue, reps=20000, seed=140033):
    cost=np.asarray(cost,dtype=float);rescue=np.asarray(rescue,dtype=float)
    assert cost.ndim==rescue.ndim==1 and cost.shape==rescue.shape
    assert len(cost)>0 and np.isfinite(cost).all() and np.isfinite(rescue).all()
    if cost.mean()==0:raise ValueError('Mean cost is zero; ratio is undefined')
    rng=np.random.default_rng(seed)
    indices=rng.integers(0,len(cost),(reps,len(cost)))
    numerator=rescue[indices].mean(axis=1)
    denominator=cost[indices].mean(axis=1)
    if np.any(denominator==0):raise ValueError('Bootstrap denominator zero; do not clip or silently discard')
    ratios=numerator/denominator
    assert np.isfinite(ratios).all()
    return dict(point=float(rescue.mean()/cost.mean()),
                interval=[float(v) for v in np.quantile(ratios,[.025,.975],method='linear')],
                index_sha256=hashlib.sha256(indices.tobytes()).hexdigest(),
                denominator_min=float(denominator.min()),denominator_max=float(denominator.max()),
                ratios=ratios)


def main():
    path=ROOT/'results/summary/p14_single_result-analysis.json'
    data=path.read_bytes()
    assert data==subprocess.check_output(['git','show','74c6ceb:results/summary/p14_single_result-analysis.json'],cwd=ROOT)
    r=json.loads(data);assert r['n']==32 and r['seeds']==list(range(1200,1232))
    cost=-np.asarray(r['primary']['native_AUC_harm']['per_seed'],dtype=float)
    rescue=np.asarray(r['primary']['onset_actor_rescue']['per_seed'],dtype=float)
    assert np.all(cost>0)
    q=paired_ratio(cost,rescue)
    assert np.isclose(q['point'],r['repaired_fraction_point'],atol=1e-14,rtol=0)
    assert np.allclose(q['interval'],r['repaired_fraction_bootstrap95'],atol=1e-14,rtol=0)
    report=dict(role='audit of frozen ratio bootstrap, not a new statistical result',
        primary_analysis_sha256=hashlib.sha256(data).hexdigest(),
        n=32,bootstrap_draws=20000,bootstrap_rng_seed=140033,quantile_method='linear',
        paired_numerator_denominator_rows=True,clipped_to_zero_one=False,
        point=q['point'],CI95=q['interval'],joint_index_sha256=q['index_sha256'],
        positive_bootstrap_denominator_min=q['denominator_min'],positive_bootstrap_denominator_max=q['denominator_max'],
        uncertainty_covered='across training seeds conditional on fixed shared evaluation tape10000, setting, reference and horizon',
        uncertainty_not_covered=['redrawn evaluation tape','task/architecture/hyperparameter selection','different horizon/reference'],
        second_key54321='prespecified separate robustness check; same training seeds, not pooled or counted twice')
    out=ROOT/'results/reproduction/p14';out.mkdir(parents=True,exist_ok=True)
    (out/'uncertainty-audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: joint paired20,000-draw bootstrap exactly reproduces the frozen91.8% ratio interval; no evaluation-marginal uncertainty claimed')


if __name__=='__main__':main()
