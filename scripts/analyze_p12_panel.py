"""Phase 12 generalization panel: per-test surgery analysis + pre-registered cross-setting summary.

For each Phase 12 test (tag p12-<name>), runs analyze_p8_surgery.py with --fork-eval and the pre-registered
TOST margin (10% of |CCC mean AUC| in that setting), then parses every per-setting analysis (earlier settings
included) into one cross-setting table.

Usage: python scripts/analyze_p12_panel.py
Output: results/summary/p12_panel-analysis.md (+ per-test p12_<name>-analysis.md)
"""
from __future__ import annotations

import glob
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

S = Path("results/summary")
TESTS = [  # pre-registered order
    ("G1", "cartpole-rflip", "CartPole-v1", "reward flip"),
    ("G2", "acrobot-rflip", "Acrobot-v1", "reward flip"),
    ("G3", "si-actnoise", "SpaceInvaders", "actuator noise p=0.5"),
    ("G4", "freeway-rflip", "Freeway-MinAtar", "reward flip"),
    ("G5", "si-obsnoise", "SpaceInvaders", "observation noise sigma=1"),
    ("G6", "si-extinction", "SpaceInvaders", "reward extinction"),
]
EARLIER = [  # settings analysed in Phases 8-11 (same script, same rules)
    ("main", "p8_ppo_surgery_n12", "SpaceInvaders", "movement restriction", 12),
    ("J3", "p8_asterix_surgery", "Asterix", "movement restriction", 8),
    ("J4", "p8_rflip_surgery", "SpaceInvaders", "reward flip", 8),
    ("J7", "p8_h256_surgery", "SpaceInvaders (2x256)", "movement restriction", 8),
    ("E2", "p9_asterix_rflip_surgery_n8", "Asterix", "reward flip", 8),
]
NUM = r"([+-]?\d+\.\d+)"
CI = NUM + r" \[" + NUM + r", " + NUM + r"\]"


def parse(md: Path) -> dict:
    t = md.read_text()
    out = {"file": md.name}
    m = re.search(r"Seeds with all factorial arms \(n = (\d+)\)", t)
    out["n"] = int(m.group(1)) if m else None
    for key, lab in [("total", "total DDD - CCC"), ("actor", "actor main effect"), ("critic", "critic main effect"),
                     ("opt", "opt main effect")]:
        m = re.search(r"\| " + re.escape(lab) + r" \| " + CI, t)
        if m:
            out[key] = tuple(float(x) for x in m.groups())
        row = re.search(r"\| " + re.escape(lab) + r" \|(.*)\n", t)
        if row and key != "total":
            cells = [c.strip() for c in row.group(1).split("|")]
            out[key + "_tost"] = cells[-2] if cells[-1] == "" else cells[-1]
            out[key + "_holm"] = cells[-3] if cells[-1] == "" else cells[-2]
    for c in ("actor", "critic", "opt"):
        m = re.search(r"\| " + c + r" \| " + CI + r"\*? \| ([+-]?\d+)% \| (\d+)/(\d+) \|", t)
        if m:
            out[c + "_share"] = int(m.group(4))
            out[c + "_seeds"] = f"{m.group(5)}/{m.group(6)}"
    m = re.search(r"Pre-registered replication criterion: \*\*([A-Z ]+)\*\*", t)
    out["verdict"] = m.group(1).strip() if m else None
    m = re.search(r"\| CCC \| " + NUM, t)
    out["ccc_auc"] = float(m.group(1)) if m else None
    m = re.search(r"raw gap at removal DDD - CCC \(\+0\): " + CI, t)
    out["gap"] = tuple(float(x) for x in m.groups()) if m else None
    m = re.search(r"C at removal: " + NUM, t)
    out["c_w1"] = float(m.group(1)) if m else None
    m = re.search(r"C at onset u\d+: " + NUM, t)
    out["c_w0"] = float(m.group(1)) if m else None
    if out["c_w1"] is None:  # fallback: clean continuation at +0 (= C at removal)
        m = re.search(r"\| CCC \| " + NUM + r" \| " + NUM, t)
        out["c_w1"] = float(m.group(2)) if m else None
    if out["gap"] is None:  # fallback: total row, +0 column
        m = re.search(r"\| total DDD - CCC \| " + CI + r"\*? \| " + CI, t)
        out["gap"] = tuple(float(x) for x in m.groups()[3:]) if m else None
    m = re.search(r"delay-adjusted AUC [^:]*: " + CI, t)
    out["delay_auc"] = tuple(float(x) for x in m.groups()) if m else None
    m = re.search(r"delay-adjusted end [^:]*: " + CI, t)
    out["delay_end"] = tuple(float(x) for x in m.groups()) if m else None
    return out


def f(ci, d=1):
    if not ci:
        return "–"
    e, lo, hi = ci
    star = "*" if (lo > 0 or hi < 0) else ""
    return f"{e:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]{star}"


def main():
    rows = []
    for gid, name, env, deficit in TESTS:
        summ = sorted(glob.glob(str(S / f"*p12-{name}-*.json")))
        done = [Path(f"/home/user/p12logs/{name}-s{c}.done").exists() for c in (0, 4)]
        if summ and not all(done) and "--include-partial" not in sys.argv:  # no interim looks (Phase 12 rule)
            rows.append((gid, env, deficit, None, "pending (not analysed until all chunks finish)"))
            continue
        if not summ:
            rows.append((gid, env, deficit, None, "not run"))
            continue
        ccc = []
        for p in summ:
            d = json.loads(Path(p).read_text())
            if "CCC" in d["arms"]:
                ccc += np.asarray(d["arms"]["CCC"]["eval_return_mean"], float)[:, 1:].mean(1).tolist()
        sesoi = round(0.1 * abs(float(np.mean(ccc))), 3)
        tag = f"p12_{name.replace('-', '_')}"
        cmd = [sys.executable, "scripts/analyze_p8_surgery.py", "--summaries", *summ, "--tag", tag, "--fork-eval",
               "--sesoi", str(sesoi), "--title",
               f"Phase 12 {gid} (pre-registered): {env}, {deficit}. TOST margin = 10% of |CCC AUC| = ±{sesoi}."]
        subprocess.run(cmd, check=True, capture_output=True)
        r = parse(S / f"{tag}-analysis.md")
        r["sesoi"] = sesoi
        status = "complete" if r["n"] == 8 else f"partial (n = {r['n']}; pre-registered 8)"
        rows.append((gid, env, deficit, r, status))
    earlier = [(gid, env, deficit, parse(S / f"{tag}-analysis.md"), f"earlier (n = {n})") for gid, tag, env, deficit, n in EARLIER
               if (S / f"{tag}-analysis.md").exists()]

    L = ["# Phase 12 — generalization panel: cross-setting summary (pre-registered)", "",
         "PPO 2^3 actor x critic x Adam surgery at removal, fork reference, unmasked/undeficited CRN evaluation. "
         "Rules (LOG Phase 12 pre-registration): damage to localize iff total CI < 0; localization criterion met iff "
         "actor main-effect CI < 0 and actor has the largest Shapley share; critic/Adam TOST at ±10% of |CCC AUC| "
         "(earlier settings: ±3 return points). Paired 95% t-CIs; negative = harm.", "",
         "| id | env | deficit | status | n | gap at removal (% of C) | total AUC | actor | critic | Adam | actor share | "
         "critic TOST p | Adam TOST p | delay-adj. AUC | delay-adj. end | verdict |",
         "|" + "---|" * 16]
    damaged, shares = [], []
    for gid, env, deficit, r, status in earlier + rows:
        if r is None:
            L.append(f"| {gid} | {env} | {deficit} | {status} |" + " – |" * 12)
            continue
        gap_pct = (f" ({100 * r['gap'][0] / r['c_w1']:+.0f}%)" if r.get("gap") and r.get("c_w1") else "")
        if r.get("gap") and r.get("c_w1") and r["c_w1"] < 0:
            gap_pct = " (neg. scale)"
        L.append(f"| {gid} | {env} | {deficit} | {status} | {r['n']} | {f(r.get('gap'))}{gap_pct} | {f(r.get('total'))} | "
                 f"{f(r.get('actor'))} | {f(r.get('critic'))} | {f(r.get('opt'))} | {r.get('actor_share', '–')}% "
                 f"({r.get('actor_seeds', '–')}) | {r.get('critic_tost', '–')} | {r.get('opt_tost', '–')} | "
                 f"{f(r.get('delay_auc'))} | {f(r.get('delay_end'))} | {r.get('verdict') or '–'} |")
        if r.get("total") and r["total"][2] < 0:
            damaged.append(gid)
            largest = max(("actor", "critic", "opt"), key=lambda c: r.get(c + "_share", -999))
            shares.append((gid, r.get("actor_share"), largest))
    # pre-registered addendum: out-of-sample test of the post-hoc gap-damage hypothesis
    base = {"Acrobot-v1": -500.0}  # u0 control of the long control (negative-return task); 0 elsewhere
    G = ["", "## Pre-registered addendum: gap-damage prediction (relative gap < 25% => no damage; >= 60% => damage)", "",
         "| id | setting | relative gap | prediction | observed (total CI < 0?) | hit |", "|---|---|---|---|---|---|"]
    hits, tot, pts = 0, 0, []
    for gid, env, deficit, r, status in earlier + rows:
        if r is None or not r.get("gap") or r.get("c_w1") is None or not r.get("total"):
            continue
        rel = -r["gap"][0] / (r["c_w1"] - base.get(env, 0.0))
        dmg = r["total"][2] < 0
        reltot = r["total"][0] / (r["c_w1"] - base.get(env, 0.0))
        pts.append((rel, reltot))
        if gid.startswith("G"):
            pred = "no damage" if rel < 0.25 else ("damage" if rel >= 0.60 else "none")
            hit = "–" if pred == "none" else ("yes" if (pred == "damage") == dmg else "NO")
            if pred != "none":
                tot += 1
                hits += hit == "yes"
            G.append(f"| {gid} | {env}, {deficit} | {100 * rel:.0f}% | {pred} | {'damage' if dmg else 'no damage'} | {hit} |")
    G.append("")
    G.append(f"Hits: **{hits} of {tot}** settings with a prediction.")
    if len(pts) >= 4:
        from scipy import stats
        rho = stats.spearmanr([p[0] for p in pts], [p[1] for p in pts]).statistic
        G.append(f"Descriptive: Spearman rho(relative gap, relative total damage) over {len(pts)} PPO settings = {rho:+.2f} "
                 "(negative = larger gap, more damage). Relative total = total AUC damage / C(w1) (range for Acrobot).")
    L += G
    L += ["", "## Pre-registered cross-setting summary", "",
          f"- Settings with damage to localize (total CI < 0): {len(damaged)} — {', '.join(damaged)}",
          f"- Actor is the largest carrier in {sum(1 for _, _, c in shares if c == 'actor')} of {len(shares)}: "
          + ", ".join(f"{g} {s}%" for g, s, _ in shares),
          f"- Actor Shapley share range: {min(s for _, s, _ in shares)}–{max(s for _, s, _ in shares)}%" if shares else "",
          "", "Per-test details: `p12_<name>-analysis.md`. Earlier settings: the files listed in the table."]
    (S / "p12_panel-analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
