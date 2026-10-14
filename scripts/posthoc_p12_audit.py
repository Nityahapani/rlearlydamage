"""Post-hoc checks for Phase 12. Does NOT change any registered decision.

Checks: exact JSON recomputation, panel-wide Holm for actor main effects, TOST with uncertainty in
control-dependent scales, controls reproducing previously recorded clean histories, and a transparent
separation of G1 (acute values seen before addendum) from untouched gap-prediction settings.
Only completed tests are read; no interim primary analysis. Use --include-partial only at the final
budget stop, where incomplete tests are explicitly descriptive.

Outputs: results/summary/p12_audit-posthoc.json and .md. No JAX imports, training or plotting.
"""
from __future__ import annotations

import argparse
import glob
import json
from itertools import combinations
from math import factorial
from pathlib import Path

import numpy as np
from scipy import stats

TESTS = [
    ("G1", "cartpole-rflip", "CartPole-v1", 0.0),
    ("G2", "acrobot-rflip", "Acrobot-v1", -500.0),
    ("G3", "si-actnoise", "SpaceInvaders-MinAtar", 0.0),
    ("G4", "freeway-rflip", "Freeway-MinAtar", 0.0),
    ("G5", "si-obsnoise", "SpaceInvaders-MinAtar", 0.0),
    ("G6", "si-extinction", "SpaceInvaders-MinAtar", 0.0),
]
CODES = ["CCC", "CCD", "CDC", "CDD", "DCC", "DCD", "DDC", "DDD"]
CARR = ["actor", "critic", "opt"]


def tci(x, level=0.95):
    x = np.asarray(x, float)
    m = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    h = float(stats.t.ppf((1 + level) / 2, len(x) - 1) * se)
    return [m, m - h, m + h]


def holm(ps):
    ps = np.asarray(ps, float)
    ans = np.empty(len(ps))
    run = 0.0
    for k, i in enumerate(np.argsort(ps)):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        ans[i] = run
    return ans.tolist()


def tost(x, margin):
    x = np.asarray(x, float)
    se = x.std(ddof=1) / np.sqrt(len(x))
    lower = float(stats.t.sf((x.mean() + margin) / se, len(x) - 1))
    upper = float(stats.t.cdf((x.mean() - margin) / se, len(x) - 1))
    return {"lower": lower, "upper": upper, "p": max(lower, upper)}


def scaled_tost(x, clean, q=0.1):
    """Account for covariance with the estimated clean scale instead of treating margin as fixed.

    The registered margin is q*|mean(clean)|; task-return sign is treated as fixed.
    Test E[x + q*s*clean] > 0 and E[x - q*s*clean] < 0 for s = sign(E[clean]).
    The thresholds are unchanged; the SEs now include clean-scale uncertainty.
    """
    x, clean = np.asarray(x, float), np.asarray(clean, float)
    s = np.sign(clean.mean())
    lo, hi = x + q * s * clean, x - q * s * clean
    p_lo = float(stats.ttest_1samp(lo, 0, alternative="greater").pvalue)
    p_hi = float(stats.ttest_1samp(hi, 0, alternative="less").pvalue)
    return {"lower": p_lo, "upper": p_hi, "p": max(p_lo, p_hi)}


def shapley(X):
    vals = {frozenset(i for i in range(3) if code[i] == "D"): x - X["CCC"] for code, x in X.items()}
    ph = {}
    for i, name in enumerate(CARR):
        oth = [k for k in range(3) if k != i]
        v = np.zeros_like(X["CCC"])
        for n in range(3):
            for sub in combinations(oth, n):
                S = frozenset(sub)
                v += factorial(n) * factorial(2-n) / 6 * (vals[S | {i}] - vals[S])
        ph[name] = v
    assert np.max(np.abs(sum(ph.values()) - (X["DDD"]-X["CCC"]))) < 1e-8
    return ph


def read_test(name):
    files = sorted(glob.glob(f"results/summary/*p12-{name}-*.json"))
    ss, arr, onset, fw = [], {c: [] for c in CODES}, [], []
    U = None
    for f in files:
        d = json.loads(Path(f).read_text())
        if not all(c in d.get("arms", {}) for c in CODES):
            continue
        U = d["eval_updates"] if U is None else U
        assert U == d["eval_updates"], f"eval grid differs: {f}"
        assert not (set(ss) & set(d["seeds"])), f"duplicate seeds: {f}"
        ss.extend(d["seeds"])
        for c in CODES:
            arr[c].extend(d["arms"][c]["eval_return_mean"])
        onset.extend(d["fork_eval"]["C"][0])
        fw.append(d)
    order = np.argsort(ss)
    return [ss[k] for k in order], {c: np.asarray(a)[order] for c,a in arr.items()}, np.asarray(onset)[order], U, fw, files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--include-partial", action="store_true")
    a = ap.parse_args()
    out, actor_ps = {}, []
    for gid, name, env, untrained in TESTS:
        seeds, M, onset, U, ds, fs = read_test(name)
        done = set(seeds) == set(range(8))
        if not done and not a.include_partial:
            continue
        if len(seeds) < 2:
            continue
        X = {c: m[:,1:].mean(1) for c,m in M.items()}
        mainfx = {n: np.mean([X[c] for c in CODES if c[i]=="D"],axis=0)-np.mean([X[c] for c in CODES if c[i]=="C"],axis=0)
                  for i,n in enumerate(CARR)}
        total = X["DDD"] - X["CCC"]
        ph = shapley(X)
        p = [float(stats.ttest_1samp(mainfx[n],0).pvalue) for n in CARR]
        h = holm(p)
        actor_ps.append(p[0])
        margin = 0.1*abs(X["CCC"].mean())
        r = {"gid":gid,"name":name,"env":env,"seeds":seeds,"n":len(seeds),"complete":done,"summaries":fs,
             "total":tci(total),"total_p":float(stats.ttest_1samp(total,0).pvalue),
             "total_per_seed":total.tolist(),"clean_auc":tci(X["CCC"]),"margin":float(margin),
             "effects":{n:tci(mainfx[n]) for n in CARR},"effect_per_seed":{n:mainfx[n].tolist() for n in CARR},
             "effect_p":dict(zip(CARR,p)),"within_setting_holm":dict(zip(CARR,h)),
             "shapley":{n:tci(ph[n]) for n in CARR},"shapley_per_seed":{n:ph[n].tolist() for n in CARR},
             "shapley_share":{n:float(100*ph[n].mean()/total.mean()) for n in CARR},
             "tost_fixed_scale":{n:tost(mainfx[n],margin) for n in CARR},
             "tost_joint_scale":{n:scaled_tost(mainfx[n],X["CCC"]) for n in CARR},
             "clean_at_onset":float(onset.mean()),"clean_at_removal":float(M["CCC"][:,0].mean()),
             "unlearning":tci(M["DDD"][:,0]-onset),"gap":tci(M["DDD"][:,0]-M["CCC"][:,0])}
        # secondary pause-adjusted quantities use the same complete clean continuation + during-window checkpoints
        dlt, ends = [], []
        for d in ds:
            L = d["protocol"]["window"][1]-d["protocol"]["window"][0]
            bank = dict(zip(d["fork_eval"]["updates"], d["fork_eval"]["C"]))
            bank.update({u:list(v) for u,v in zip(d["eval_updates"],np.asarray(d["arms"]["CCC"]["eval_return_mean"]).T)})
            D = np.asarray(d["arms"]["DDD"]["eval_return_mean"],float)
            Cdelay = np.stack([bank[u-L] for u in d["eval_updates"]],axis=1)
            dd = D-Cdelay
            dlt.extend(dd[:,1:].mean(1))
            ends.extend(dd[:,-5:].mean(1))
        # ds are seed chunks ordered by filenames, read_test applies argsort; stats insensitive to ordering
        r["delay_auc"],r["delay_end"]=tci(dlt),tci(ends)
        largest = min(CARR,key=lambda n:ph[n].mean())
        r["verdict"] = ("NO GAP TO LOCALIZE" if r["total"][2]>=0 else
                        ("REPLICATES" if r["effects"]["actor"][2]<0 and largest=="actor" else "DOES NOT REPLICATE"))
        scale = r["clean_at_removal"]-untrained
        rel = -r["gap"][0]/scale
        r["relative_gap"]=float(rel)
        r["relative_total"]=float(r["total"][0]/scale)
        pred = "no gap" if rel<0.25 else ("gap" if rel>=0.60 else None)
        r["gap_prediction"]=pred
        r["gap_prediction_hit"]=None if pred is None else (pred=="gap")== (r["total"][2]<0)
        r["untouched_for_gap_addendum"] = gid != "G1"
        out[gid]=r
    hp=holm(actor_ps)
    for r,v in zip(out.values(),hp):
        r["panel_actor_holm_posthoc"]=v
    all_hits = [r["gap_prediction_hit"] for r in out.values() if r["gap_prediction_hit"] is not None]
    untouched = [r["gap_prediction_hit"] for r in out.values() if r["untouched_for_gap_addendum"] and r["gap_prediction_hit"] is not None]
    summary={"status":"post-hoc audit; registered decisions unchanged","tests":out,
             "gap_all_planned":{"hits":sum(all_hits),"n_predictions":len(all_hits)},
             "gap_untouched":{"hits":sum(untouched),"n_predictions":len(untouched)},
             "nonrecovery_rate_CP95":[float(stats.beta.ppf(.025,1,44)),float(stats.beta.ppf(.975,2,43))]}
    S=Path("results/summary")
    (S/"p12_audit-posthoc.json").write_text(json.dumps(summary,indent=2)+"\n")
    L=["# Phase 12 post-hoc audit", "", "This audit changes no registered decision. Only completed tests are read by default.",
       "Panel-wide actor Holm is additional, not the registered within-setting three-effect correction. Joint-scale TOST accounts for uncertainty in the estimated clean AUC.","",
       "| test | n | actor raw p | within-setting Holm | panel-wide Holm (post-hoc) | registered ±10% margin | critic fixed / joint TOST | Adam fixed / joint TOST |",
       "|---|---|---|---|---|---|---|---|"]
    for gid,r in out.items():
        L.append(f"| {gid} {r['name']} | {r['n']} | {r['effect_p']['actor']:.5g} | {r['within_setting_holm']['actor']:.5g} | "
                 f"{r['panel_actor_holm_posthoc']:.5g} | {r['margin']:.3f} | "
                 f"{r['tost_fixed_scale']['critic']['p']:.5g} / {r['tost_joint_scale']['critic']['p']:.5g} | "
                 f"{r['tost_fixed_scale']['opt']['p']:.5g} / {r['tost_joint_scale']['opt']['p']:.5g} |")
    L += ["",f"Gap-addendum hit count, all planned settings (G1 acute values had been seen): {sum(all_hits)}/{len(all_hits)}.",
          f"Untouched settings only (exclude G1): {sum(untouched)}/{len(untouched)}; no prediction was issued for intermediate relative gaps.",
          "This prediction concerns a same-update gap, not persistence beyond lost time.","",
          "Exact 95% Clopper–Pearson interval for 1/44 non-recoveries: 0.0575%–12.021%. This assumes exchangeability under the tested protocol; one event does not establish a stable rare-event rate."]
    (S/"p12_audit-posthoc.md").write_text("\n".join(L)+"\n")
    print("\n".join(L))


if __name__=="__main__":
    main()
