"""Emit LaTeX rows for the paper's generalization tables from the Phase 12 per-test analyses (no hand-copied numbers).

Usage: python scripts/make_p12_tables.py  -> results/summary/p12_tables.tex
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("panel", "scripts/analyze_p12_panel.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
S = Path("results/summary")
LABEL = {"cartpole-rflip": "CartPole, reward flip", "acrobot-rflip": "Acrobot, reward flip",
         "si-actnoise": "SpaceInvaders, actuator noise", "freeway-rflip": "Freeway, reward flip",
         "si-obsnoise": "SpaceInvaders, observation noise", "si-extinction": "SpaceInvaders, reward extinction"}
BASE = {"acrobot-rflip": -500.0}


def ci(t, d=1, star=True):
    e, lo, hi = t
    s = "\\sig" if star and (lo > 0 or hi < 0) else ""
    return f"${e:+.{d}f}\\ci{{{lo:+.{d}f}}}{{{hi:+.{d}f}}}{s}$"


def pt(t, d=1):
    e, lo, hi = t
    return f"${e:+.{d}f}" + ("\\sig$" if (lo > 0 or hi < 0) else "$")


def main():
    gen, gap = [], []
    for gid, name, env, deficit in P.TESTS:
        md = S / f"p12_{name.replace('-', '_')}-analysis.md"
        chunks = (0, 2, 4, 6) if name == "freeway-rflip" else (0, 4)
        done = all(Path(f"/home/user/p12logs/{name}-s{c}.done").exists() for c in chunks)
        if not (md.exists() and done):
            continue
        r = P.parse(md)
        d = 0 if abs(r["total"][0]) >= 20 else 1
        dmg = r["total"][2] < 0
        verdict = (f"{r['actor_share']}\\%; replicates" if r.get("verdict") == "REPLICATES" else
                   ("n/a; no gap to localize" if not dmg else f"{r.get('actor_share', '–')}\\%; not met"))
        gen.append(f"{LABEL[name]} ({gid}) & {r['n']} & {ci(r['gap'], d)} & {ci(r['total'], d)} & "
                   f"{ci(r['delay_auc'], d) if r.get('delay_auc') else '--'} & {pt(r['actor'], d)} & {pt(r['critic'], d)} & "
                   f"{pt(r['opt'], d)} & {verdict} \\\\")
        scale = r["c_w1"] - BASE.get(name, 0.0)
        relgap = -r["gap"][0] / scale
        pred = "no gap" if relgap < 0.25 else ("gap" if relgap >= 0.60 else "no prediction")
        obs = "gap" if dmg else "no gap"
        hit = obs if pred == "no prediction" else obs + (" (hit)" if (pred == "gap") == dmg else " (miss)")
        gap.append(f"{LABEL[name]} ({gid}) & {r['n']} & {100 * relgap:.0f}\\% & {ci(r['total'], d)} & "
                   f"${100 * r['total'][0] / scale:+.0f}\\%$ & {ci(r['delay_auc'], d) if r.get('delay_auc') else '--'} & "
                   f"{pred}; {hit} \\\\")
    out = ["% tab:general rows (Phase 12)"] + gen + ["", "% tab:gap rows (Phase 12; prediction; hit)"] + gap
    (S / "p12_tables.tex").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
