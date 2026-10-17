"""Software-only counterexamples guard paired ratio semantics; never scientific seed rows."""
from pathlib import Path
import importlib.util
import json

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('ratio_audit',ROOT/'scripts/verify_p14_uncertainty.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def test_frozen_joint_interval_reproduces():
    r=json.loads((ROOT/'results/summary/p14_single_result-analysis.json').read_text())
    q=mod.paired_ratio(-np.asarray(r['primary']['native_AUC_harm']['per_seed']),
                       r['primary']['onset_actor_rescue']['per_seed'])
    assert np.allclose(q['interval'],r['repaired_fraction_bootstrap95'],atol=1e-14,rtol=0)
    assert q['denominator_min']>0


def test_proportional_pairs_keep_covariance_and_no_clipping():
    cost=np.array([1.,2.,3.,4.])
    q=mod.paired_ratio(cost,1.5*cost,reps=1000)
    assert np.allclose(q['ratios'],1.5,atol=1e-14,rtol=0)
    assert q['interval'][0]>1  # a bootstrap percentage is not silently capped at100%


def test_ratio_of_means_not_mean_of_ratios():
    q=mod.paired_ratio([1.,3.],[1.,1.],reps=1000)
    assert q['point']==.5
    assert not np.isclose(q['point'],np.mean(np.array([1.,1.])/np.array([1.,3.])))


def test_zero_denominator_requires_explicit_failure():
    with pytest.raises(ValueError,match='zero'):
        mod.paired_ratio([0.,0.],[1.,1.],reps=10)
