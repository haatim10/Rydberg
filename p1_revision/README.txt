PAPER 1 REVIEWER REVISION

Purpose
Address the review of ThesisProj-7.pdf without altering the historical Paper 1
implementation, verified campaign, or Paper 2 experiments. This module lives
on the existing codex branch. No claim of global convergence, optimal
shrinkage, hardware validation or guaranteed per-trial improvement is made.

Protocol
The full configuration list is defined in scenarios.configurations and frozen
in results/p1_revision/campaign/manifest.json before campaign outcomes are
analyzed. Each operating cell has 400 trials.

1. Structured baselines: L=3,7,14 at SNR=0,10 dB; N32/K3/P30/RSR12.
   Six methods: EM-GS, HS-GS, Rician GD, Rician GD+Hankel, EM-GS+smooth
   shrinkage, and accurately solved Xu 1D strong-reference linearized LS.
   All observations are generated with the exact nonlinear magnitude model.
   The three structured methods select their own scalar rank/threshold on
   the same held-out pilots, then refit all pilots. Every user column is
   processed separately using the shared selected scalar. No true channel,
   true path count or true effective rank enters selection.
   6 cells, 2400 independent worlds.

2. Held-out fraction: 10/20/30/40 percent, P10/30, SNR0/10; N32/K3/RSR12,
   independent per-user path counts 3 to 7. Final estimates always use all P.
   Pilot folds are nested trailing columns, matching original P1 at 30 percent.
   Gaussian pilot columns are exchangeable. A common world is reused across
   fractions. 16 cells, 6,400 condition-trial evaluations, 1,600 base worlds.

3. Position mismatch: Gaussian position-error standard deviations
   0/.005/.01/.02 wavelengths, L3/12, N32/K3/P30/SNR5/RSR12.
   Positions are n/2+std*u_n wavelengths. Same channel paths, pilots,
   underlying standard normal position offsets and noise draw are used
   across error strengths. The actual perturbed channel is the target.
   The known reference is calibrated at the actual positions (unchanged
   for P1's broadside reference). Statistical power/SNR conventions are
   unchanged; no realized-channel renormalization is introduced.
   These perturbation levels are stress tests, not hardware specifications.
   8 cells, 3,200 evaluations, 800 base channel/pilot/noise draws.

Total: 30 cells, 12,000 condition-trial evaluations, 4,800 independent base-world
groups. Position perturbations change G and Z within a group, so these are
7,200 distinct observation sets, with pilot-fraction reuse accounting for
the remaining repetitions. Do not call all 12,000 independent Monte Carlo
worlds. Pairing is retained for statistical comparisons.

Classical settings
100 final and 25 candidate updates, 4 structural sweeps per application.
GD uses the exact Rician Wirtinger derivative with packed-real convention
grad=2D and eta=sigma2/(2||S||_2^2). EM uses a right linear solve.
Smooth filtering: sigma -> sigma^3/(sigma^2+tau^2), where tau is one of
0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6 times current per-user Hankel RMS singular value.
The scale is recomputed each sweep. Zero factor is an exact bypass.
This is an empirical smooth-filter control, not OptShrink or an exact
proximal step for a vector-domain Hankel nuclear norm. Rank candidates
span 1 through ceil(N/2); full rank is exact EM/GD bypass. All tuning minimizes
the held-out sum of squared amplitude residuals. No test-truth tuning.
Equal update counts do not imply equal computation or global convergence.

Reproduction from repository root
Use Python with numpy, scipy, matplotlib installed. No neural-network runtime
is required. Set OPENBLAS_NUM_THREADS, OMP_NUM_THREADS, MKL_NUM_THREADS,
VECLIB_MAXIMUM_THREADS, NUMEXPR_NUM_THREADS to 1 before Python starts, and
MPLCONFIGDIR to a writable cache directory.

  python -m unittest discover -s p1_revision -t . -p 'test_*.py' -v
  python -m p1_revision.run --preflight
  python -m p1_revision.run --workers 4 --chunk 5 --max-hours 4

After COMPLETE.json exists:
  python -m p1_revision.analyze
  python -m p1_revision.audit

Additional completed studies:
  python -m p1_revision.reuse rsr
  python -m p1_revision.reuse summary
  python -m p1_revision.summarize_budget

The first command reanalyses the included classical RSR/noise errors. The
second reanalyses saved EM proxy replays. The third audits and summarises
the saved 100/300-update trajectories. They do not launch the main campaign.
Use a working copy of the release when regenerating summaries: output files
and audit timestamps are overwritten. Preserve the delivered evidence.

To regenerate the optional studies from observations rather than summaries:
  python -m p1_revision.reuse proxy --scope all
  python -m p1_revision.budget_check
  python -m p1_revision.serial_timing

These commands resume existing outputs. Move to a separate output directory
or working copy for genuinely fresh computation; do not erase the delivered
results. Serial timings can change between executions and machines.

Portability note: the frozen main-campaign runner records the SHA256 of
/Users/haatim/Downloads/ThesisProj-7.pdf when it freezes a campaign. A fresh
freeze therefore expects that local document path. Auditing and analysing
the completed results do not need the PDF. Porting this metadata path would
change the runner hash and requires a new manifest, not a silent resume of
the frozen campaign. Dependency versions and platform are in each protocol.

Each chunk is written atomically. Source hashes and configurations must
match on resume; incompatible results are rejected. A source snapshot is
stored beside the manifest. Trial count and compute cap are independent
of observed estimator gains. A cap produces INCOMPLETE.json, not a finished
campaign. Running the same command resumes verified completed chunks.

Statistics and timing
NMSE is 10 log10(sum squared error / sum channel energy). Method gain is
10log10(sum reference error / sum method error). Paired median is a
different, separately reported statistic. Pointwise percentile 95 percent
intervals use 4,000 paired world bootstrap resamples with seed 20261007.
No simultaneous coverage or universal superiority is implied.
Parallel campaign timings are workload measurements, not serial speed
benchmarks. Any dedicated serial timing results must be labeled separately.

Reuse and manuscript files
p1_revision/reuse.py audits and reanalyzes classical RSR/noise-variance
results from the separate audited sahure_expansion campaign, and replays
only EM-GS on existing verified P1 worlds to obtain observable-rank proxies.
It does not add those worlds to the original 27,300-trial P1 count.
Results and explicit provenance are in results/p1_revision/reused/.
Insertable LaTeX and exact paragraph anchors are in
paper/paper1/reviewer_revision/. The latest supplied full TeX predates the
reviewed PDF, so the revision pack does not silently overwrite or pretend
to rebuild ThesisProj-7 from an older source.
