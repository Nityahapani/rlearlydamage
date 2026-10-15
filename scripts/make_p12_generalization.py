"""Figure for the Phase 12 expanded PPO generalization panel.

Recomputes carrier and total contrasts directly from committed per-seed JSONs.
All effects are scaled by mean clean AUC (Acrobot: clean AUC + 500, competence above the untrained floor).
This visualization is not a new primary analysis; no pooled cross-task inference is performed.
Only complete n=8 Phase 12 settings are plotted. Also includes the five prior settings.

Outputs: figures/p12_generalization.png/.pdf; results/summary/p12_figure-source.json.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from analyze_p12_panel import EARLIER, TESTS, ready
from posthoc_p12_audit import CODES, CARR, shapley, tci

S=Path("results/summary")
NAMES={
    "main":"SpaceInvaders · movement",
    "J3":"Asterix · movement",
    "J4":"SpaceInvaders · reward flip",
    "J7":"SpaceInvaders · wider MLP",
    "E2":"Asterix · reward flip",
    "G1":"CartPole · reward flip",
    "G2":"Acrobot · reward flip",
    "G3":"SpaceInvaders · actuator noise",
    "G4":"Freeway · reward flip",
    "G5":"SpaceInvaders · observation noise",
    "G6":"SpaceInvaders · extinction",
}
COLORS={"actor":"#196D67","critic":"#8168AA","opt":"#C27E38"}


def load(paths,untrained=0):
    data={c:{} for c in CODES}
    bank={}
    U=None
    for p in paths:
        d=json.loads(Path(p).read_text())
        u=d["eval_updates"]
        U=u if U is None else U
        assert U==u,"eval grids differ"
        for c in CODES:
            if c not in d["arms"]:
                continue
            for i,s in enumerate(d["seeds"]):
                x=np.asarray(d["arms"][c]["eval_return_mean"][i],float)
                if s in data[c]:
                    assert np.allclose(data[c][s],x,atol=.001),"duplicate trajectories disagree"
                data[c][s]=x
        fk=d.get("fork_eval")
        if fk:
            for i,s in enumerate(d["seeds"]):
                b=dict(zip(fk["updates"],np.asarray(fk["C"])[:,i]))
                if "CCC" in d["arms"]:
                    b.update(dict(zip(u,d["arms"]["CCC"]["eval_return_mean"][i])))
                bank[s]=b
    ss=sorted(set.intersection(*(set(data[c]) for c in CODES)))
    ev={c:np.stack([data[c][s] for s in ss]) for c in CODES}
    A={c:m[:,1:].mean(1) for c,m in ev.items()}
    fx={n:np.mean([A[c] for c in CODES if c[i]=="D"],axis=0)-np.mean([A[c] for c in CODES if c[i]=="C"],axis=0)
        for i,n in enumerate(CARR)}
    total=A["DDD"]-A["CCC"]
    ph=shapley(A)
    scale=float(A["CCC"].mean()-untrained)
    assert scale>0
    dlt=None
    if bank and all(s in bank for s in ss):
        # Window length is fixed within a setting. Infer from the first source with fork evaluations.
        d0=next(json.loads(Path(p).read_text()) for p in paths if json.loads(Path(p).read_text()).get("fork_eval"))
        w0,w1=d0["protocol"]["window"]
        L=w1-w0
        delayed=np.asarray([[bank[s][u-L] for u in U[1:]] for s in ss])
        dlt=ev["DDD"][:,1:].mean(1)-delayed.mean(1)
    gap=tci(total)
    return {"n":len(ss),"seeds":ss,"scale":scale,"sources":[str(p) for p in paths],
            "total_raw":gap,"total_pct":tci(total/scale*100),
            "delay_raw":tci(dlt) if dlt is not None else None,
            "delay_pct":tci(dlt/scale*100) if dlt is not None else None,
            "effects_raw":{n:tci(x) for n,x in fx.items()},
            "effects_pct":{n:tci(x/scale*100) for n,x in fx.items()},
            "actor_share":float(ph["actor"].mean()/total.mean()*100) if gap[2]<0 else None}


def main():
    rows=[]
    for gid,tag,env,deficit,n in EARLIER:
        md=S/f"{tag}-analysis.md"
        line=next(l for l in md.read_text().splitlines() if l.startswith("Summaries: "))
        paths=[S/f.strip() for f in line.split(": ",1)[1].split(",")]
        r=load(paths)
        r.update(id=gid,label=NAMES[gid],phase="prior")
        rows.append(r)
    for gid,name,env,deficit in TESTS:
        if not ready(name):
            continue
        paths=[Path(f) for f in sorted(glob.glob(str(S/f"*p12-{name}-*.json")))
               if all(c in json.loads(Path(f).read_text()).get("arms",{}) for c in CODES)]
        r=load(paths,-500 if gid=="G2" else 0)
        r.update(id=gid,label=NAMES[gid],phase="Phase 12")
        rows.append(r)
    (S/"p12_figure-source.json").write_text(json.dumps({"purpose":"visualization; no new primary decision","settings":rows},indent=2)+"\n")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.spines.left":False,"axes.titleweight":"bold","axes.titlepad":12})
    fig,(a,b)=plt.subplots(1,2,figsize=(12.8,max(5.8,len(rows)*.53+2.0)),sharey=True,
                          gridspec_kw={"width_ratios":[1.12,1]},constrained_layout=True)
    for j,r in enumerate(rows):
        if j%2==0:
            for ax in (a,b):ax.axhspan(j-.49,j+.49,color="#F4F5F6",zorder=0)
        for k,n in enumerate(CARR):
            y=j+[-.20,0,.20][k]
            m,lo,hi=r["effects_pct"][n]
            a.errorbar(m,y,xerr=[[m-lo],[hi-m]],fmt="o",color=COLORS[n],capsize=2.4,markersize=5,
                       elinewidth=1.4,label={"actor":"Actor","critic":"Critic","opt":"Adam"}[n] if j==0 else None)
        for key,off,color,mark,label in [("total_pct",-.12,"#30343B","s","Same-update"),
                                         ("delay_pct",.12,"#2477AE","o","Pause-relative")]:
            if r[key] is None:continue
            m,lo,hi=r[key]
            b.errorbar(m,j+off,xerr=[[m-lo],[hi-m]],fmt=mark,color=color,capsize=2.4,markersize=5,
                       elinewidth=1.4,label=label if j==1 else None)
    # Prior vs new settings are separated; outcomes for null settings remain visible.
    for ax in (a,b):
        ax.axhline(4.5,color="#BDC3C8",lw=1)
        ax.axvline(0,color="#69727A",lw=1,ls="--")
        ax.grid(axis="x",color="#E6E9EB",lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("% of clean competence",labelpad=9)
        ax.tick_params(axis="y",length=0)
    a.set_yticks(np.arange(len(rows)),[f"{r['label']}  (n={r['n']})" for r in rows],fontsize=14)
    a.invert_yaxis()
    a.set_title("A  Component effects")
    b.set_title("B  Recovery cost")
    a.legend(loc="upper left",bbox_to_anchor=(0,-.16),frameon=False,ncol=1,fontsize=13)
    b.legend(loc="upper left",bbox_to_anchor=(0,-.16),frameon=False,ncol=1,fontsize=13)
    fig.savefig("figures/p12_generalization.png",dpi=210,facecolor="white")
    fig.savefig("figures/p12_generalization.pdf",facecolor="white")
    print(f"Saved generalization figure ({len(rows)} settings); raw numeric sources in p12_figure-source.json")


if __name__=="__main__":
    main()
