"""Write m2_timing/ADDENDUM.md from the follow-up result files (no hand-copied numbers).

  .venv/bin/python -m experiments.final_filter_single_c.m2_timing.addendum
  .venv/bin/python -m experiments.final_filter_single_c.report     # appends it to REPORT.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = 10.74
SCOPES = (('full_grid', 'Full grid'), ('snr_30_60', '30–60 dB (all RSR)'), ('rsr5_above_30', 'RSR 5 > 30 dB'))


def f(x, d=2, sign=False):
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def timing_table(t, verdict):
    out = ['| Arm | ' + ' | '.join(f'{lab} p90 / max ms' for _, lab in SCOPES) + ' | p90 ratio vs A (full / 30–60 / RSR 5) |' +
           (' 10.74 ms: full grid and 30–60 | 10.74 ms: RSR 5 > 30 |' if verdict else ''),
           '| --- ' * (len(SCOPES) + 2 + (2 if verdict else 0)) + '|']
    for arm, v in t['summary'].items():
        ratio = '—' if arm == 'A' else ' / '.join(f"{t['p90_ratio_vs_A'][arm][k]:.3f}" for k, _ in SCOPES)
        row = f'| {arm} | ' + ' | '.join(f"{v[k]['p90_ms']:.2f} / {v[k]['max_ms']:.1f}" for k, _ in SCOPES) + f' | {ratio} |'
        if verdict:
            cap = v['absolute_cap']
            row += f" {'pass' if cap['full_grid_and_snr_30_60'] else '**fail**'} | {'pass' if cap['rsr5_above_30'] else '**fail**'} |"
        out.append(row)
    return '\n'.join(out)


def main():
    s = json.loads((HERE/'results/followup_c3_summary.json').read_text())
    cloud = json.loads((HERE/'results/timing_cloud_x86_diagnostic.json').read_text())
    m2_path = HERE/'results/timing_m2.json'
    m2 = json.loads(m2_path.read_text()) if m2_path.exists() else None
    ag = s['aggregates']
    labels = dict(grid='Full grid (153)', grid_30_60='30–60 dB (81)', offgrid_snr='Off-grid SNR, grid RSR (45)',
                  **{'snr31.25': '31.25 dB, all RSR (18)'}, rsr0='RSR 0 (66)', rsr8='RSR 8 (66)', rsr16='RSR 16 (66)')
    acc = ['| Scope | A | C2 | C3 | C2 − A [95% CI] | C3 − A [95% CI] | C3 − C2 [95% CI] |', '| --- ' * 7 + '|']
    for k, lab in labels.items():
        v = ag[k]
        d = lambda key: f"{f(v['diff'][key]['point'], sign=True)} [{f(v['diff'][key]['ci95'][0], sign=True)}, {f(v['diff'][key]['ci95'][1], sign=True)}]"
        acc.append(f"| {lab} | {f(v['mean_setting_nmse_db']['A'])} | {f(v['mean_setting_nmse_db']['C2'])} | "
                   f"{f(v['mean_setting_nmse_db']['C3'])} | {d('C2-A')} | {d('C3-A')} | {d('C3-C2')} |")
    at31 = [r for r in s['rows'] if r['snr'] == 31.25 and r['rsr'] in (12., 16., 20.)]
    t31 = ['| Cell | EM tail | C2 − A dB | C3 − A dB |', '| --- | --- | --- | --- |'] + [
        f"| `{r['cell']}` | {r['em_tail']} | {f(r['diff']['C2-A']['point'], sign=True)} | {f(r['diff']['C3-A']['point'], sign=True)} |"
        for r in at31]
    worse = s['Q2_cells_C3_worse_than_C2_by_over_0p3dB']
    worst = max(worse, key=lambda r: r['c3_minus_c2']) if worse else None
    lim = s['limits_descriptive']
    q1 = s['Q1_31p25']
    if m2 is None:
        m2_text = ('**Pending: not yet run on the Apple M2.** This session runs on a cloud x86 machine (Intel Xeon, 4 cores) '
                   'and could not reach the M2. Run on the M2, from the repository root with nothing else running:\n\n'
                   '```bash\n.venv/bin/python -m experiments.final_filter_single_c.m2_timing.timing_interleaved --tag m2\n```\n\n'
                   'Then regenerate this addendum and the report:\n\n'
                   '```bash\n.venv/bin/python -m experiments.final_filter_single_c.m2_timing.addendum\n```\n\n'
                   '```bash\n.venv/bin/python -m experiments.final_filter_single_c.report\n```\n\n'
                   'Until then, **no pass/fail against 10.74 ms exists for C2 or C3 on the reference hardware.**')
    else:
        m2_text = (f"Machine: {m2['environment']['cpu']} ({m2['environment']['platform']}). Load average before/after: "
                   f"{', '.join(f'{x:.2f}' for x in m2['load_average_before'])} / {', '.join(f'{x:.2f}' for x in m2['load_average_after'])}. "
                   f"{m2['calls_per_arm']} timed calls per arm.\n\n" + timing_table(m2, True))
    text = f"""
---

# ADDENDUM (post hoc, 31 October 2026): M2 timing and arm C3

**Everything in this addendum was planned after the main test results were seen.**
It does not change the frozen results, choices or verdicts above. Pre-registration
for the addendum: [m2_timing/configs/followup_preregistration.json](m2_timing/configs/followup_preregistration.json),
committed before any follow-up data. Code and results: [m2_timing/](m2_timing/).

## A1. Runtime limit and method

- **The limit is the original pre-registered one:** absolute p90 ≤ 10.74 ms, one CPU
  thread, double precision. The 1.02× arm-A ratio in Section 4 was only the
  fallback for the cloud machine, where arm A itself exceeded 10.74 ms.
- **Method:** arms A, C2 and C3 run interleaved in one process. Each world and fit
  calls all three back to back on the same inputs, with the order rotated per
  call. There is a warm-up and nothing else runs. Each arm makes 1,377 calls
  (459 grid worlds × 3 fits). Online outputs are checked to equal the offline
  arm computation exactly at the first world of every setting.
- **Scopes.** The follow-up request calls the second scope the "30–60 dB target
  range" (all RSR). The main pre-registration defined the target range as RSR 5
  above 30 dB. Both are reported. The pass/fail columns apply the limit to the
  full grid together with 30–60 dB, and separately to RSR 5 above 30 dB.

## A2. Apple M2 timing

{m2_text}

## A3. Cloud diagnostic (not an M2 result, no pass/fail)

The same interleaved script on the cloud machine ({cloud['environment']['cpu'].replace('model name	: ', '')}):

{timing_table(cloud, False)}

Interleaving removes most of the drift between the two separate timing windows in
Section 6. The paired per-call ratio against A has a median of
{cloud['paired_call_ratio_vs_A']['C2']['p50_ms']:.4f} for C2 and
{cloud['paired_call_ratio_vs_A']['C3']['p50_ms']:.4f} for C3. On this machine A's p90 is already close to
10.74 ms at 30–60 dB, so these absolute values say nothing about the M2.

## A4. Arm C3: filter only where the EM tail is nonzero

C3 is C2 (c = 4, β = 1, nominal SNR > 30 dB), filtering only where the routed
EM-tail length is > 0. That means reference key 5 from 40 dB, reference keys 12/20
from 32.5 dB, and never at 31.25 dB. It was evaluated on fresh draws (seed
2026103121): 396 cells × 128 channels × 3 fits, with the main study's metrics and
20,000-draw block bootstrap. Arm A again reproduced the frozen receiver exactly
in every cell.

Mean-setting NMSE (dB) and differences:

{chr(10).join(acc)}

**Q1. Does C3 remove the > 1 dB losses at 31.25 dB? Yes.**
- C3 is identical to the table at 31.25 dB (difference exactly 0).
- On these fresh draws, C2 was > 1 dB worse than A in {len(q1['c2_cells_over_1db'])} cells at 31.25 dB
  ({', '.join(f'`{c}`' for c in q1['c2_cells_over_1db'])}).
- C3 has none.

{chr(10).join(t31)}

**Q2. Does C3 keep C2's gains elsewhere? No.**
- C3 is worse than C2 by more than 0.3 dB in {len(worse)} cells.
- All of them are at reference key 5 (RSR 0, 5 and 8) between 31.25 and 38.75 dB.
  There the receiver's EM tail is zero, so C3 stops filtering where C2 gained most.
- The worst is `{worst['cell'] if worst else '—'}`, where C3 is {f(worst['c3_minus_c2'], sign=True) if worst else '—'} dB worse than C2.
- Elsewhere C3 ≈ C2. At RSR 16, C3 is slightly better because it skips 31.25 dB.

**Against the table (main-study limits, descriptive because post hoc):**
- C3: L1 {f(lim['L1_full'], sign=True)} dB full grid and {f(lim['L1_30_60'], sign=True)} dB at 30–60 dB.
  L2 worst grid setting {f(lim['L2_worst_db'], sign=True)} dB (`{lim['L2_worst']}`).
  L3: {len(lim['L3_cells'])} settings with BER harm.
- On the same fresh draws, C2's worst grid setting is {f(lim['L2_C2_worst_db'], sign=True)} dB, with
  {len(lim['L3_C2_cells'])} BER-harm settings.
- BER is identical across A, C2 and C3 in every scope, and zero at 30–60 dB.

![C2 and C3 minus A](m2_timing/figures/followup_c3_vs_c2.png)

**Reading.**
- The 31.25 dB losses do not come from the filter strength. They come from the
  frozen receiver's routing at that point: global start, zero-length tail, and a
  30 dB table key. Removing the filter wherever the tail is zero fixes 31.25 dB at
  RSR 12/16/20, but also removes the large RSR 5 gains between 32.5 and 38.75 dB.
- On these draws, C2 is ahead of C3 by {f(ag['grid']['diff']['C3-C2']['point'])} dB over the full grid and
  {f(ag['grid_30_60']['diff']['C3-C2']['point'])} dB at 30–60 dB.
- So whether C2 replaces the table still depends on the M2 timing (A2). C3 is the
  more conservative variant if you want no setting to lose to the table at all.
- A narrower variant that skips the filter only at the 31.25 dB mismatch, or a fix
  to that routing quirk, was not tested.
"""
    (HERE/'ADDENDUM.md').write_text(text.lstrip('\n'))
    print('ADDENDUM.md written; M2 timing', 'present' if m2 else 'pending')


if __name__ == '__main__':
    main()
