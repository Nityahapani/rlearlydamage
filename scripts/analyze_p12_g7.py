"""Apply Phase 12 G7 registered decisions to PQN-R reward-flip network x buffer surgery.

Primary: epsilon=.05 eval, mean over +10..+650 updates, paired 95% t-CIs; rules in bf9a946.
Secondary registered before G7 data (3ddffe2): +20,+40,...,+300, matching the PPO panel's 15 checkpoints
in environment steps. This does not replace the original primary.
Additional post-hoc sensitivity: Holm over network/buffer main effects, and covariance-aware scaled TOST.
No primary interim look: requires both completed chunks unless --include-partial is explicitly requested at
budget end, in which case the output is labelled descriptive only.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats
from posthoc_p12_audit import holm, scaled_tost, tci, tost

METRIC = "eval_eps05_return_mean"
CODES = ["CC", "DC", "CD", "DD"]


def fmt(x):
    m,lo,hi=tci(x)
    star="*" if lo>0 or hi<0 else ""
    return f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]{star}"


def contrasts(sc):
    return {
        "total DD - CC":sc["DD"]-sc["CC"],
        "network main":(sc["DC"]+sc["DD"]-sc["CC"]-sc["CD"])/2,
        "buffer main":(sc["CD"]+sc["DD"]-sc["CC"]-sc["DC"])/2,
        "network transfer DC - CC":sc["DC"]-sc["CC"],
        "data transfer CD - CC":sc["CD"]-sc["CC"],
        "network rescue CD - DD":sc["CD"]-sc["DD"],
        "buffer rescue DC - DD":sc["DC"]-sc["DD"],
        "interaction":sc["DD"]-sc["CD"]-sc["DC"]+sc["CC"],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--include-partial",action="store_true")
    a=ap.parse_args()
    files=sorted(glob.glob("results/summary/*p12-pqnr-rflip-*.json"))
    ds=[json.loads(Path(f).read_text()) for f in files]
    ds=[d for d in ds if all(c in d.get("arms",{}) for c in CODES)]
    if not ds:
        raise SystemExit("No complete G7 factorial chunk")
    U=np.asarray(ds[0]["eval_updates"])
    seeds=sum((d["seeds"] for d in ds),[])
    assert len(seeds)==len(set(seeds)), "duplicate seed"
    done=set(seeds)==set(range(8))
    if not done and not a.include_partial:
        raise SystemExit("G7 not complete: primary analysis withheld until n=8 (no formal interim primary look)")
    assert all(d["eval_updates"]==U.tolist() for d in ds)
    order=np.argsort(seeds)
    seeds=[seeds[k] for k in order]
    ev={c:np.concatenate([np.asarray(d["arms"][c][METRIC],float) for d in ds],axis=0)[order] for c in CODES}
    k=U-U[0]
    pri={c:m[:,1:].mean(1) for c,m in ev.items()}
    matched_ix=np.flatnonzero((k>=20)&(k<=300)&(k%20==0))
    assert len(matched_ix)==15, "incomplete matched recovery grid"
    mat={c:m[:,matched_ix].mean(1) for c,m in ev.items()}
    fx,mfx=contrasts(pri),contrasts(mat)
    margin=float(0.1*abs(pri["CC"].mean()))
    ps={name:float(stats.ttest_1samp(x,0).pvalue) for name,x in fx.items()}
    hp=holm([ps["network main"],ps["buffer main"]])
    total,net,buf=(tci(fx[n]) for n in ["total DD - CC","network main","buffer main"])
    damage=total[2]<0
    net_criterion=net[2]<0 and abs(net[0])>abs(buf[0])
    buf_criterion=buf[2]<0
    b_tost=tost(fx["buffer main"],margin)
    b_joint=scaled_tost(fx["buffer main"],pri["CC"])
    partial=not(done and len(seeds)==8)
    status="DESCRIPTIVE ONLY: incomplete planned n=8" if partial else "Registered G7 primary (n=8), bf9a946"
    result={"status":status,"seeds":seeds,"n":len(seeds),"summaries":files,"metric":METRIC,
            "protocol":ds[0]["protocol"],"primary_grid":k[1:].tolist(),"matched_grid":k[matched_ix].tolist(),
            "arm_auc":{c:tci(x) for c,x in pri.items()},"contrasts":{n:tci(x) for n,x in fx.items()},
            "p":ps,"matched_contrasts":{n:tci(x) for n,x in mfx.items()},
            "per_seed":{n:x.tolist() for n,x in fx.items()},
            "buffer_margin":margin,"buffer_TOST":b_tost,"buffer_joint_scale_TOST_posthoc":b_joint,
            "Holm_two_effects_posthoc":{"network":hp[0],"buffer":hp[1]},
            "same_update_gap_detected":bool(damage),"network_criterion_met":bool(net_criterion),
            "buffer_criterion_met":bool(buf_criterion)}
    if damage:
        result["shares"]={n:float(fx[n+" main"].mean()/fx["total DD - CC"].mean()*100) for n in ["network","buffer"]}
    L=["# Phase 12 G7 — PQN-R reward-flip network × buffer surgery", "",status,"",
       f"Seeds: {seeds}; epsilon=.05; deficit [250,350), primary recovery +10..+650 (2.662M steps).",
       "Network = parameters + RAdam state; buffer = all buffer arrays. Environment/obs/RNG/counter from C. Wrong rewards are stored during the restriction window; evaluation always uses the true reward.","",
       "| contrast | primary AUC, 95% t-CI | primary p | matched-horizon AUC (secondary), 95% t-CI |",
       "|---|---|---|---|"]
    for name,x in fx.items():
        L.append(f"| {name} | {fmt(x)} | {ps[name]:.6g} | {fmt(mfx[name])} |")
    L += ["", "## Registered primary decisions", "",
          f"- Same-update gap detected (DD−CC upper CI <0): {damage}.",
          f"- Network carrier criterion (negative main effect; larger magnitude than buffer): {net_criterion}.",
          f"- Buffer carrier criterion (negative main effect): {buf_criterion}.",
          f"- Buffer TOST at ±{margin:.3f} (10% of |CC AUC|): p={b_tost['p']:.6g}.",
          f"- Shares (only if total gap detected): {result.get('shares','not interpretable: total includes zero')}.",
          "", "## Matched horizon (registered secondary, 3ddffe2)","",
          "PQN-R +20,+40,...,+300 exactly matches PPO +10,+20,...,+150 in environment steps: 15 checkpoints over 1,228,800 recovery steps. Deficit duration is also matched (409,600 steps). This is not the original G7 primary.","",
          "## Additional sensitivity (post-hoc)","",
          f"Holm across network/buffer: network {hp[0]:.6g}, buffer {hp[1]:.6g}. Covariance-aware control-scaled buffer TOST p={b_joint['p']:.6g}.","",
          "## Overwrite diagnostics", "",
          "| arm | first post-removal buffer fraction | fraction at +25 | first update with fraction zero | mean sampled window-era fraction |",
          "|---|---|---|---|---|"]
    overw={}
    for c in CODES:
        B=np.concatenate([np.asarray(d["arms"][c]["train"]["buf_frac_window"],float) for d in ds],axis=0).mean(0)
        P=np.concatenate([np.asarray(d["arms"][c]["train"]["sample_frac_window"],float) for d in ds],axis=0).mean(0)
        zero=next((i+1 for i,v in enumerate(B) if v<=0),None)
        overw[c]={"first_fraction":float(B[0]),"fraction_at_25":float(B[24]),"zero_update":zero,"mean_sample_fraction":float(P.mean())}
        L.append(f"| {c} | {B[0]:.4f} | {B[24]:.4f} | {zero} | {P.mean():.4f} |")
    result["overwrite"]=overw
    S=Path("results/summary")
    (S/"p12_g7_pqnr_rflip-analysis.json").write_text(json.dumps(result,indent=2)+"\n")
    (S/"p12_g7_pqnr_rflip-analysis.md").write_text("\n".join(L)+"\n")
    print("\n".join(L))


if __name__=="__main__":
    main()
