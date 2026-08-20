"""Focused sweep inside the per-ray-polarization branch (the only lever that moved)."""
import numpy as np
from zf_sweep import evaluate, SCEN, CUI_ZF_TARGET_DB

BASE = dict(geom="ULA-x", per_elem_pol=False, cui_ang=True, scen="UMa-NLOS",
            three_d=False, row_norm=True, mu_axis="y")
cur = evaluate(**dict(BASE, per_elem_pol=True))
print(f"current (per-element pol): ZF={cur['zf_db']:.2f} dB   target {CUI_ZF_TARGET_DB:.1f} dB\n")

rows = []


def add(label, n=400, **over):
    kw = dict(BASE); kw.update(over)
    r = evaluate(n=n, **kw); r["label"] = label
    r["shift"] = r["zf_db"] - cur["zf_db"]
    rows.append(r)
    print(f"{label:<40} ZF={r['zf_db']:7.2f} shift={r['shift']:+6.2f} "
          f"cond={r['cond']:6.2f} corr={r['corr']:.3f}")


print("--- per-ray pol x geometry ---")
for g in ("ULA-x", "ULA-y", "UPA-6x6", "UPA-12x3", "UPA-18x2"):
    add(f"per-ray, geom={g}", geom=g)

print("\n--- per-ray pol x scenario (Cui angle override kept) ---")
for s in SCEN:
    add(f"per-ray, scen={s}", scen=s)

print("\n--- per-ray pol x 38.901 eq.(18) angles ---")
for s in SCEN:
    add(f"per-ray, eq.18, {s}", cui_ang=False, scen=s)

print("\n--- per-ray pol x eq.18 angles x 3D ---")
for s in ("UMa-NLOS", "UMi-NLOS", "RMa-NLOS", "InH-NLOS"):
    add(f"per-ray, eq.18+3D, {s}", cui_ang=False, scen=s, three_d=True)

print("\n--- dipole orientation within per-ray branch ---")
add("per-ray, mu||x", mu_axis="x")
add("per-ray, ULA-y, mu||y", geom="ULA-y", mu_axis="y")

best = sorted(rows, key=lambda r: abs(r["zf_db"] - CUI_ZF_TARGET_DB))[:8]
print(f"\n=== CLOSEST TO TARGET ({CUI_ZF_TARGET_DB:.1f} dB) ===")
for r in best:
    print(f"  {r['label']:<42} ZF={r['zf_db']:7.2f}  shift={r['shift']:+6.2f}")
