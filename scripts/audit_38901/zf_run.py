import numpy as np
from zf_sweep import evaluate, SCEN, CUI_ZF_TARGET_DB

BASE = dict(geom="ULA-x", per_elem_pol=True, cui_ang=True, scen="UMa-NLOS",
            three_d=False, row_norm=True, mu_axis="y")

ref = evaluate(**BASE)
print(f"CURRENT PRODUCTION MODEL: Tr={ref['tr']:.5f}  ZF@12dB={ref['zf_db']:.2f} dB "
      f"cond={ref['cond']:.2f} corr={ref['corr']:.3f}")
print(f"TARGET (Cui, pixel-extracted): ZF@12dB ~ {CUI_ZF_TARGET_DB:.1f} dB "
      f"(need ~{CUI_ZF_TARGET_DB-ref['zf_db']:+.1f} dB)\n")

rows = []


def add(label, **over):
    kw = dict(BASE); kw.update(over)
    r = evaluate(**kw)
    r["label"] = label
    r["shift"] = r["zf_db"] - ref["zf_db"]
    rows.append(r)
    print(f"{label:<34} Tr={r['tr']:.5f}+-{r['tr_se']:.5f} ZF={r['zf_db']:7.2f} "
          f"shift={r['shift']:+6.2f} cond={r['cond']:6.2f} corr={r['corr']:.3f}"
          + (f"  [{r['bad']} rank-deficient]" if r["bad"] else ""))


print("--- A. ARRAY GEOMETRY (Cui specifies only N=36) ---")
for g in ("ULA-x", "ULA-y", "UPA-6x6", "UPA-4x9", "UPA-12x3", "UPA-18x2", "UPA-2x18"):
    add(f"geom={g}", geom=g)

print("\n--- B. GEOMETRY x 3D angles (UPA needs zenith; Cui gives none) ---")
for g in ("ULA-x", "UPA-6x6", "UPA-4x9", "UPA-12x3"):
    add(f"geom={g}, 3D", geom=g, three_d=True)

print("\n--- C. DIPOLE ORIENTATION (mu_eg vs array axis) ---")
add("ULA-x, mu||x (endfire null)", mu_axis="x")
add("ULA-y, mu||y (endfire null)", geom="ULA-y", mu_axis="y")
add("UPA-6x6, mu||x, 3D", geom="UPA-6x6", mu_axis="x", three_d=True)

print("\n--- D. POLARIZATION CONVENTION ---")
add("per-ray pol (38.901 eq.10)", per_elem_pol=False)
add("per-ray pol + UPA-6x6 3D", per_elem_pol=False, geom="UPA-6x6", three_d=True)

print("\n--- E. ROW NORMALIZATION ---")
add("no row normalization", row_norm=False)
add("no row norm + UPA-6x6 3D", row_norm=False, geom="UPA-6x6", three_d=True)

print("\n--- F. 38.901 POWER-DERIVED ANGLES (drop Cui U(-90,90) override) ---")
for s in SCEN:
    add(f"eq.18 angles, {s}", cui_ang=False, scen=s, three_d=True)

print("\n--- G. SCENARIO r_tau/zeta only (Cui angle override kept) ---")
for s in SCEN:
    add(f"scen={s}", scen=s)

best = sorted(rows, key=lambda r: abs(r["zf_db"] - CUI_ZF_TARGET_DB))[:6]
print(f"\n=== CLOSEST TO CUI TARGET ({CUI_ZF_TARGET_DB:.1f} dB) ===")
for r in best:
    print(f"  {r['label']:<36} ZF={r['zf_db']:7.2f}  shift={r['shift']:+6.2f}")
