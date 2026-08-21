"""Content of the Rydberg atomic-MIMO technical report."""
from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages

from make_report_pdf import FIGS, OUT, Doc


def build() -> None:
    with PdfPages(OUT) as pdf:
        d = Doc(pdf)

        # ---------------------------------------------------------- title --
        d.new_page()
        d.y = 0.72
        d.fig.text(0.5, 0.80, "Atomic MIMO Receivers", ha="center",
                   fontsize=26, weight="bold")
        d.fig.text(0.5, 0.755, "Formulae, Methods and Results",
                   ha="center", fontsize=15, color="0.3")
        d.fig.add_artist(__import__("matplotlib").pyplot.Line2D(
            [0.28, 0.72], [0.735, 0.735], color="0.6", lw=1.0))
        d.fig.text(0.5, 0.700,
                   "Reproduction of Cui et al., \"Towards Atomic MIMO Receivers\"\n"
                   "(IEEE JSAC 43(3), March 2025)\n\n"
                   "and an exact-model ULA channel-estimation study",
                   ha="center", fontsize=11, color="0.2", linespacing=1.7)
        d.fig.text(0.5, 0.55,
                   "Track A — detection on the Cui 3GPP TR 38.901 channel\n"
                   "Track B — channel estimation on a geometric ULA channel",
                   ha="center", fontsize=10, color="0.35", linespacing=1.8)
        d.fig.text(0.5, 0.36,
                   "Every equation in this report was transcribed from the\n"
                   "implementation and verified numerically by an independent\n"
                   "audit (141 checks, scripts/deep_audit.py).",
                   ha="center", fontsize=9, color="0.45", linespacing=1.7,
                   style="italic")

        # ------------------------------------------------------ Part I: model
        d.new_page("1  The observation model")
        d.para(
            "Both tracks share one physical model. A Rydberg atomic receiver "
            "measures the magnitude of the total electric field at each sensor. "
            "The phase is destroyed by the measurement, which is what makes "
            "detection a phase-retrieval problem rather than a linear inverse "
            "problem.")
        d.h2("1.1  Detection (Track A)")
        d.para("For one channel use, with $N$ atomic sensors and $K$ single-antenna users:")
        d.eq(r"\mathbf{z} \;=\; \bigl|\, \mathbf{A}^{\mathsf H}\mathbf{s} "
             r"\;+\; \mathbf{b} \;+\; \mathbf{w} \,\bigr|", "(1)")
        d.bullets([
            r"$\mathbf{A}\in\mathbb{C}^{K\times N}$ — known channel (Cui eq. 16)",
            r"$\mathbf{s}\in\mathbb{C}^{K}$ — unknown QAM symbols, $\mathbb{E}|s_k|^2=1$",
            r"$\mathbf{b}\in\mathbb{C}^{N}$ — known local-oscillator reference",
            r"$\mathbf{w}\sim\mathcal{CN}(0,\sigma^2\mathbf{I})$ — additive noise, "
            r"inside the magnitude",
            r"$|\cdot|$ is the elementwise amplitude — $|\mathcal{E}|$, never $|\mathcal{E}|^2$",
        ])
        d.h2("1.2  Channel estimation (Track B)")
        d.para(
            "The unknown becomes the channel and the pilots are known. Over $P$ "
            "pilot instants:")
        d.eq(r"\mathbf{Z} \;=\; \bigl|\, \mathbf{G}\mathbf{S} \;+\; \mathbf{B} "
             r"\;+\; \mathbf{W} \,\bigr|", "(2)")
        d.para(
            r"with $\mathbf{G}\in\mathbb{C}^{N\times K}$ unknown, "
            r"$\mathbf{S}\in\mathbb{C}^{K\times P}$ known pilots, and "
            r"$\mathrm{vec}(\mathbf{W})\sim\mathcal{CN}(0,\sigma^2\mathbf{I})$. "
            r"This is the exact nonlinear model. No strong-reference "
            r"linearisation is used in any Track-B estimator reported here; a "
            r"runtime tripwire asserts the linearised solver is never called.")
        d.h2("1.3  Canonical form")
        d.para(
            "Both reduce to one canonical problem, which is why a single "
            "validated solver serves both. Per receive element $n$:")
        d.eq(r"\mathbf{z} \;=\; \bigl|\,\mathbf{M}^{\mathsf H}\mathbf{u} + "
             r"\mathbf{b} + \mathbf{w}\,\bigr|", "(3)")
        d.table([
            ["", "detection", "channel estimation"],
            [r"$\mathbf{M}$", r"$\mathbf{A}$", r"$\overline{\mathbf{S}}$ (conjugate pilots)"],
            [r"$\mathbf{u}$", r"$\mathbf{s}$", r"$\overline{\mathbf{g}_n}$ (conjugate row)"],
            [r"$\mathbf{z}$", r"$\mathbf{z}$", r"$\mathbf{Z}_{n,:}$"],
        ], [0.10, 0.24, 0.40])
        d.para(
            "The channel-estimation adapter loops over receive elements, calls "
            "the same solver with $\\mathbf{M}=\\mathbf{S}$, "
            "$\\mathbf{b}=\\overline{\\mathbf{B}_{n,:}}$, and sets "
            "$\\hat{\\mathbf{G}}_{n,:}=\\overline{\\hat{\\mathbf{u}}}$. The "
            "conjugation was verified by noiseless recovery: the estimator "
            "returns $\\mathbf{G}$ to a relative error of $2.4\\times10^{-15}$, "
            "and not $\\overline{\\mathbf{G}}$.")

        # --------------------------------------------------- Part II: channels
        d.new_page("2  Channel models")
        d.h2("2.1  Track A — Cui 3GPP TR 38.901 clustered channel")
        d.para(
            "Cui generates coefficients with the 3GPP TR 38.901 model, giving "
            "only Table I. Each user's row is a sum over clusters and rays:")
        d.eq(r"a_{n,k} \;=\; \sum_{c=1}^{N_c}\sum_{r=1}^{M_r} "
             r"\alpha_{c,r}\, e^{-\jmath 2\pi f_c \tau_c}\, "
             r"\bigl(\boldsymbol{\mu}_{eg}^{\mathsf T}\boldsymbol{\epsilon}_{n,c,r}\bigr)\,"
             r"e^{-\jmath (n-1)\pi\sin\theta_{c,r}}", "(4)")
        d.table([
            ["Table I parameter", "value"],
            ["clusters $N_c$ / rays per cluster $M_r$", "23 / 20"],
            [r"path gains $\alpha_{c,r}$", r"$\mathcal{CN}(0,1)$"],
            [r"incident angles $\theta$", r"$\mathcal{U}(-90^\circ,90^\circ)$"],
            [r"max angle spread per cluster", r"$\mathcal{U}(-5^\circ,5^\circ)$"],
            [r"max delay spread", r"$\mathcal{U}(0,30\,\mathrm{ns})$"],
            [r"carrier $f_c$", r"$5$ GHz"],
        ], [0.42, 0.30])
        d.para(
            r"Rows are normalised so $\mathrm{mean}_n|a_{n,k}|^2=1$, which is "
            r"what makes eqs. (6)–(7) exact. Cui $\S$VI-A samples "
            r"$\boldsymbol{\epsilon}_{n,c,r}$ per antenna element; that choice "
            r"whitens the array response and is the origin of the documented "
            r"$\approx 2$ dB offset in Figs. 5–6 (Section 7).")
        d.h2("2.2  Track B — geometric ULA channel")
        d.para(
            "A half-wavelength ULA with resolvable specular paths. For user $k$:")
        d.eq(r"\psi_{\ell,k} = \pi\sin\theta_{\ell,k}, \qquad "
             r"g_{n,k} \;=\; \sum_{\ell=1}^{L_k}\alpha_{\ell,k}\,"
             r"e^{-\jmath (n-1)\psi_{\ell,k}}", "(5)")
        d.para(
            r"equivalently $\mathbf{g}_k=\mathbf{A}(\boldsymbol{\theta}_k)"
            r"\boldsymbol{\alpha}_k$ with the array manifold "
            r"$\mathbf{a}(\theta)=[1,e^{-\jmath\psi},\dots,e^{-\jmath(N-1)\psi}]^{\mathsf T}$, "
            r"$\|\mathbf{a}\|_2^2=N$. Parameters: "
            r"$\theta_{\ell,k}\sim\mathcal{U}[-\pi/2,\pi/2]$, "
            r"$\alpha_{\ell,k}\sim\mathcal{CN}(0,\beta_k/L_k)$ so that "
            r"$\mathbb{E}|g_{n,k}|^2=\beta_k$, and $L_k\sim\mathcal{U}\{3,\dots,7\}$ "
            r"drawn independently per user per realisation.")

        # ------------------------------------------------- Part III: calibration
        d.new_page("3  SNR and RSR calibration")
        d.para(
            "These two definitions set every operating point in every figure. "
            "Both were re-derived from first principles during the audit and "
            "checked against the implementation.")
        d.h2("3.1  SNR — Cui eq. (36)")
        d.eq(r"\mathrm{SNR} \;=\; \frac{\mathbb{E}\bigl(|\mathbf{a}_n^{\mathsf H}"
             r"\mathbf{s}|^2\bigr)}{\mathbb{E}\bigl(|w_n|^2\bigr)}", "(6)")
        d.para(
            r"With row-normalised $\mathbf{A}$, unit-energy QAM and independent "
            r"users, the numerator is $\sum_k \mathbb{E}|a_{n,k}|^2\mathbb{E}|s_k|^2 = K$. "
            r"Hence the noise variance actually used is")
        d.eq(r"\sigma^2 \;=\; \frac{K}{\mathrm{SNR}_{\mathrm{lin}}}, \qquad "
             r"\mathrm{SNR}_{\mathrm{lin}} = 10^{\mathrm{SNR_{dB}}/10}", "(7)")
        d.para(
            "Note this is total signal power over all $K$ users, so fixing the "
            "SNR does not fix the per-user SNR: doubling $K$ at fixed SNR "
            "doubles $\\sigma^2$.")
        d.h2("3.2  RSR — Cui eq. (37)")
        d.eq(r"\mathrm{RSR} \;=\; \frac{\mathbb{E}\bigl(|b_n|^2\bigr)}"
             r"{\mathbb{E}\bigl(|a_{n,k}s_k|^2\bigr)}", "(8)")
        d.para(
            r"The denominator is a $\textbf{single user}$'s contribution, not "
            r"the sum over $K$. This is the single easiest place to introduce a "
            r"factor-$K$ error. With $\beta_{\mathrm{ref}}=1$ and "
            r"$\mathbb{E}|s_b|^2=1$ it gives")
        d.eq(r"|\alpha_b| \;=\; \sqrt{\mathrm{RSR}_{\mathrm{lin}}}"
             r"\qquad\text{(not } \sqrt{K\,\mathrm{RSR}_{\mathrm{lin}}}"
             r"\text{, not } \sqrt{\mathrm{RSR}_{\mathrm{lin}}/K}\text{)}", "(9)")
        d.para(
            "The audit explicitly tests that the implemented value differs from "
            "both incorrect alternatives, and measures the achieved SNR and RSR "
            "empirically from generated worlds: 2.82 dB and 12.15 dB against "
            "targets of 3 and 12 dB.")

        # ------------------------------------------------- Part IV: algorithms
        d.new_page("4  Detection and estimation algorithms")
        d.h2("4.1  Spectral initialisation")
        d.para(
            "All iterative solvers start from Cui's augmented spectral "
            "initialiser. With $\\bar{\\mathbf{m}}_q=[\\mathbf{m}_q;\\,b_q]$:")
        d.eq(r"\mathbf{M}_{\mathrm{spec}} = \sum_q z_q\, \bar{\mathbf{m}}_q "
             r"\bar{\mathbf{m}}_q^{\mathsf H} \in\mathbb{C}^{(D+1)\times(D+1)},"
             r"\qquad \bar{\mathbf{u}}_0 = \text{principal eigenvector}", "(10)")
        d.h2("4.2  Biased Gerchberg–Saxton (Cui Algorithm 1)")
        d.para(
            "GS alternates between restoring the measured magnitude with the "
            "currently estimated phase, and re-solving a linear least squares. "
            "At iteration $t$:")
        d.eq(r"\boldsymbol{\lambda}^{t-1} = \mathbf{M}^{\mathsf H}\mathbf{u}^{t-1} "
             r"+ \mathbf{b}, \qquad \boldsymbol{\theta}^{t} = "
             r"\angle\boldsymbol{\lambda}^{t-1}", "(11)")
        d.eq(r"\mathbf{y}^{t} = \mathbf{z}\odot e^{\jmath\boldsymbol{\theta}^{t}},"
             r"\qquad \mathbf{r}^{t} = \mathbf{y}^{t}-\mathbf{b}", "(12)")
        d.eq(r"\bigl(\mathbf{M}\mathbf{M}^{\mathsf H}\bigr)\mathbf{u}^{t} "
             r"= \mathbf{M}\mathbf{r}^{t}", "(13)")
        d.para(
            "The measured amplitude $\\mathbf{z}$ is kept exactly; only the "
            "phase is supplied by the current iterate. Solved via the normal "
            "equations, never an explicit inverse. Cui sets $t_0=50$ iterations, "
            "as used throughout.")
        d.h2("4.3  EM-GS (Cui Algorithm 2)")
        d.para(
            "EM-GS keeps the same phase and least-squares steps but weights the "
            "restored observation by the Bessel ratio, which is the conditional "
            "mean of the Rician amplitude — a soft, SNR-aware version of the "
            "hard magnitude substitution:")
        d.eq(r"R(\kappa) = \frac{I_1(\kappa)}{I_0(\kappa)}, \qquad "
             r"\boldsymbol{\kappa} = \frac{2}{\sigma^2}\,\mathbf{z}\odot"
             r"|\boldsymbol{\lambda}|", "(14)")
        d.para(
            r"$R$ is monotone increasing with $R(0)=0$ and $R(\kappa)\to1$. That "
            r"limit explains an observed behaviour: at high SNR or high RSR "
            r"$R\to1$ and EM-GS degenerates into plain GS, so the two become "
            r"indistinguishable. The implementation evaluates $R$ with "
            r"exponentially scaled Bessel functions to avoid overflow, never as "
            r"a raw ratio $I_1/I_0$. The audit checks it against SciPy and "
            r"against the quadrature definition "
            r"$I_n(\kappa)=\frac{1}{\pi}\int_0^\pi e^{\kappa\cos t}\cos(nt)\,dt$.")
        d.h2("4.4  Genie ZF with known phase")
        d.para("Given the true noisy phase — information a magnitude-only receiver "
               "never has — the problem becomes linear:")
        d.eq(r"\mathbf{r} = \mathbf{z}\odot e^{\jmath\boldsymbol{\theta}}-\mathbf{b},"
             r"\qquad \bigl(\mathbf{M}\mathbf{M}^{\mathsf H}\bigr)\hat{\mathbf{u}} "
             r"= \mathbf{M}\mathbf{r}", "(15)")
        d.para(
            "This is a benchmark, not a valid estimator. It is allowed to lie "
            "below the CRLB of Section 4.6, because that bound applies to "
            "estimators seeing only $\\mathbf{z}$.")

        d.new_page("4  Detection and estimation algorithms (cont.)")
        d.h2("4.5  Exhaustive search — LS and ML")
        d.para("Cui's two optimal-detector benchmarks, searched over the full "
               "constellation to the power $K$:")
        d.eq(r"J_{\mathrm{LS}}(\mathbf{u}) = \bigl\|\,\mathbf{z} - "
             r"|\mathbf{M}^{\mathsf H}\mathbf{u}+\mathbf{b}|\,\bigr\|_2^2", "(16)")
        d.eq(r"J_{\mathrm{ML}}(\mathbf{u}) = \sum_q \Bigl[-\frac{|\lambda_q|^2}"
             r"{\sigma^2} + \log I_0\!\Bigl(\frac{2 z_q|\lambda_q|}{\sigma^2}"
             r"\Bigr)\Bigr]", "(17)")
        d.para(
            "LS and ML are not assumed identical: (17) is the $\\mathbf{u}$-"
            "dependent part of the exact Rician log-likelihood, evaluated in the "
            "log domain. Feasible for Fig. 7(a)/8 ($4^3=64$ candidates) but not "
            "for Fig. 7(b) ($16^6\\approx1.7\\times10^7$), which is exactly why "
            "Cui omits it there and so do we.")
        d.h2("4.6  Cramér–Rao lower bound")
        d.para("Each amplitude is Rician with density")
        d.eq(r"p(z\mid\lambda) = \frac{2z}{\sigma^2}\exp\!\Bigl(-\frac{z^2+"
             r"|\lambda|^2}{\sigma^2}\Bigr) I_0\!\Bigl(\frac{2z|\lambda|}"
             r"{\sigma^2}\Bigr)", "(18)")
        d.para("giving the Fisher information and bound")
        d.eq(r"\mathbf{F} = \sum_q \beta_q\, \mathbf{m}_q\mathbf{m}_q^{\mathsf H},"
             r"\qquad \beta_q = \frac{\mathbb{E}\bigl[z_q^2R^2(\kappa_q)\bigr]"
             r"-|\lambda_q|^2}{\sigma^4}", "(19)")
        d.para(
            r"$\beta_q$ is evaluated by numerical quadrature over a two-sided "
            r"window centred on $|\lambda|$, never clipped to its high-SNR "
            r"limit. As $\mathrm{SNR}\to\infty$, $\beta_q\to 1/(2\sigma^2)$, so "
            r"$\mathbf{F}\to\frac{1}{2\sigma^2}\mathbf{M}\mathbf{M}^{\mathsf H}$ "
            r"and the bound approaches "
            r"$2\sigma^2(\mathbf{M}\mathbf{M}^{\mathsf H})^{-1}$ — exactly "
            r"$10\log_{10}2 = 3.0103$ dB above the genie-ZF covariance. That is "
            r"the price of losing phase, and the audit measures it at "
            r"$3.0103$ dB.")

        # --------------------------------------------------- Part V: metrics
        d.new_page("5  Metrics and aggregation")
        d.para(
            "How results are pooled matters as much as how they are computed. "
            "Two rules are applied everywhere and are the most common source of "
            "silently wrong curves.")
        d.h2("5.1  NMSE — ratio of sums, never mean of ratios")
        d.eq(r"\mathrm{NMSE} = \frac{\sum_{\mathrm{trials}}\|\mathbf{s}-"
             r"\tilde{\mathbf{s}}\|_2^2}{\sum_{\mathrm{trials}}"
             r"\mathbb{E}\|\mathbf{s}\|_2^2}, \qquad "
             r"\mathrm{NMSE_{dB}} = 10\log_{10}(\mathrm{NMSE})", "(20)")
        d.para(
            r"For channel estimation the same rule uses Frobenius energies, "
            r"$\mathrm{NMSE}_G=\sum\|\hat{\mathbf{G}}-\mathbf{G}\|_F^2/"
            r"\sum\|\mathbf{G}\|_F^2$. Three points matter:")
        d.bullets([
            r"It is $10\log_{10}$, not $20\log_{10}$ — NMSE is already a power "
            r"ratio, so a factor-2 dB error is the trap here.",
            r"The denominator is the $\textit{expected}$ symbol energy $K$ for "
            r"unit-energy QAM, not a per-trial $\|\mathbf{s}\|^2$.",
            r"$\tilde{\mathbf{s}}$ is the $\textit{continuous}$ solver output, "
            r"before any constellation demapping.",
        ])
        d.para(
            "Averaging per-trial dB values instead would give a different and "
            "smaller number, because it is a geometric rather than arithmetic "
            "mean of the energies. The audit constructs a case where the two "
            "differ by 7 dB.")
        d.h2("5.2  BER — global bit errors over global bits")
        d.para(
            "The demapping step Cui describes: project each recovered symbol to "
            "the nearest constellation point, then to bits.")
        d.eq(r"\mathrm{BER} = \frac{\sum_{\mathrm{trials}}(\text{bit errors})}"
             r"{\sum_{\mathrm{trials}}(\text{bit count})}", "(21)")
        d.para(
            "Not the mean of per-trial BERs. Gray mapping means adjacent "
            "constellation points differ in exactly one bit, so a symbol error "
            "usually costs one bit of $\\log_2 M$; consequently "
            "$\\mathrm{SER}/\\log_2 M \\le \\mathrm{BER} \\le \\mathrm{SER}$. "
            "The audit verifies the Gray property by walking every "
            "constellation neighbour ($M=4,16,64$, zero violations) and "
            "recounts the bit errors independently from the constellation bit "
            "table.")
        d.h2("5.3  Zero-error points")
        d.para(
            "At high SNR some points produce no bit errors at all. A zero "
            "cannot be drawn on a log axis and must not be reported as "
            "$\\mathrm{BER}=0$. We use the one-sided Wilson score bound:")
        d.eq(r"\mathrm{BER} \;\le\; \frac{\hat p + \frac{z^2}{2n} + "
             r"z\sqrt{\frac{\hat p(1-\hat p)}{n}+\frac{z^2}{4n^2}}}"
             r"{1+\frac{z^2}{n}}, \qquad z = 1.645", "(22)")
        d.para(
            "which stays valid at $k=0$ and for small $n$, unlike the normal "
            "approximation. Such points are omitted from the plotted curve and "
            "their bounds retained in the aggregate CSV. Section 7 shows these "
            "bounds already sit below anything Cui plots, so the resolution is "
            "sufficient.")
        d.h2("5.4  Common random numbers")
        d.para(
            "At each operating point one frozen world "
            "$\\{\\mathbf{A},\\mathbf{s},\\mathbf{b},\\mathbf{w},\\mathbf{z}\\}$ "
            "is generated from a key derived from (trial, SNR, RSR) and handed "
            "to every algorithm. Comparisons are therefore paired, removing "
            "channel variability from algorithm differences, and any trial is "
            "reproducible in isolation.")

        _figures(d)
        _trackb(d)
        _audit(d)
        d.close()
    print(f"wrote {OUT}")


def _figures(d: Doc) -> None:
    d.new_page("6  Track A — reproduction of Cui's figures")
    d.para(
        "Five published figures, all on the Cui channel with the equations of "
        "Sections 3–5 and $t_0=50$ iterations. Error bars are deliberately not "
        "drawn; confidence intervals live in the aggregate CSVs.")

    d.h2("6.1  Fig. 5 — detection NMSE vs SNR")
    d.para(r"$N\times K = 36\times3$, 16-QAM, RSR $=12$ dB, 2 000 trials/SNR, "
           r"18 points. Uses eqs. (1), (4), (6)–(7), (10)–(14), (15), (19), (20).")
    d.image(FIGS / "fig5_clean.png", height=0.255)
    d.caption("Fig. 5 — detection NMSE vs SNR. Every ordering Cui reports is "
              "reproduced: EM-GS below biased GS at all SNR, both approaching "
              "the CRLB, and the CRLB sitting 3.01 dB above genie ZF at high SNR.")
    d.para(
        "The absolute level sits about 2 dB below Cui's published curve. This "
        "is a channel-conditioning effect, discussed in Section 7, not a solver "
        "or calibration error.")

    d.h2("6.2  Fig. 6 — detection NMSE vs RSR")
    d.para(r"$N\times K=36\times3$, 16-QAM, SNR $=3$ dB, 500 trials/RSR, "
           r"26 points, RSR $0\ldots25$ dB.")
    d.image(FIGS / "fig6_clean.png", height=0.255)
    d.caption("Fig. 6 — detection NMSE vs RSR. The genie-ZF curve is flat "
              "because its error $(\\mathbf{A}\\mathbf{A}^{\\mathsf H})^{-1}"
              "\\mathbf{A}\\mathbf{w}$ does not involve $\\mathbf{b}$; the "
              "fitted slope is $-0.0026$ dB/dB, not statistically significant.")
    d.para(
        "The GS/EM-GS gap shrinks as RSR grows, from $-1.77$ dB at RSR $=0$ to "
        "$-0.14$ dB at RSR $=25$, which is exactly the $R(\\kappa)\\to1$ "
        "behaviour predicted by eq. (14).")

    d.new_page("6  Track A — reproduction of Cui's figures (cont.)")
    d.h2("6.3  Fig. 7(a) — BER vs SNR, small-scale")
    d.para(r"$N\times K=36\times3$, 4-QAM, RSR $=12$ dB. 333 000 trials, "
           r"1 998 000 bits per algorithm. Adds eqs. (16), (17), (21), (22).")
    d.image(FIGS / "fig7a_clean.png", height=0.245)
    d.caption("Fig. 7(a) — BER vs SNR. Ordering matches Cui exactly: ZF, then "
              "exhaustive ML below exhaustive LS, then EM-GS below biased GS, "
              "each dashed optimal search below its iterative counterpart.")
    d.h2("6.4  Fig. 7(b) — BER vs SNR, large-scale")
    d.para(r"$N\times K=100\times6$, 16-QAM, RSR $=12$ dB. 122 000 trials, "
           r"2 928 000 bits per algorithm. Exhaustive search omitted exactly as "
           r"Cui does, since $16^6\approx1.7\times10^7$ candidates per trial is "
           r"prohibitive.")
    d.image(FIGS / "fig7b_clean.png", height=0.235)
    d.caption("Fig. 7(b) — BER vs SNR. The measured EM-GS-to-ZF SNR gap is "
              "3.79–3.97 dB across BER $3\\times10^{-2}$ to $10^{-3}$; Cui "
              "states \"between 3 $\\sim$ 4 dB\".")

    d.new_page("6  Track A — reproduction of Cui's figures (cont.)")
    d.h2("6.5  Fig. 8 — BER vs RSR")
    d.para(r"$N\times K=36\times3$, 4-QAM, SNR $=3$ dB, RSR $0\ldots25$ dB. "
           r"239 000 trials, 1 434 000 bits per algorithm.")
    d.image(FIGS / "fig8_clean.png", height=0.235)
    d.caption("Fig. 8 — BER vs RSR. Cui plots no genie-ZF curve here, so "
              "neither do we, though it is evaluated and kept in the CSV.")
    d.para(
        "Measured EM-GS improvement from RSR 0 to 20 dB is $39.1\\times$; Cui "
        "states \"more than one order of magnitude\".")

    d.h3("The caption/body contradiction in Cui's Fig. 8")
    d.para(
        "Cui's Fig. 8 caption says 16-QAM; the body text says \"the SNR is "
        "fixed as 3 dB and a 4-QAM modulator is adopted\". The two cannot both "
        "be right, so both were run and compared against the published levels "
        "over 12 matched (algorithm, RSR) points.")
    d.table([
        ["", "4-QAM (body text)", "16-QAM (caption)"],
        ["median BER ratio to Cui", "0.82", "24.12"],
        [r"median $|\log_{10}$ ratio$|$", "0.086", "1.367"],
    ], [0.30, 0.24, 0.24])
    d.para(
        "The 16-QAM variant sits about $29\\times$ above the 4-QAM curve and "
        "misses Cui's levels by more than an order of magnitude beyond "
        "RSR $=0$. The body text is correct and the caption is in error.")
    d.image(FIGS / "fig8_16qam_diagnostic.png", height=0.215)
    d.caption("Diagnostic only — Fig. 8 rerun at 16-QAM as the caption claims. "
              "Note the vertical scale: these levels are far from the published "
              "figure.")


def _trackb(d: Doc) -> None:
    d.new_page("7  Track B — exact-model ULA channel estimation")
    d.para(
        "Track B asks a different question: can the geometric structure of a "
        "ULA channel improve estimation while keeping Cui's exact nonlinear "
        "magnitude model? The baselines below are Cui's own algorithms, "
        "unmodified, applied to the ULA channel of eq. (5).")
    d.h2("7.1  Baselines B1 and B2")
    d.para(r"$N=8$ (matching Xu's $I=8$ for comparability), $K=3$, "
           r"$L_k\sim\mathcal{U}\{3,\dots,7\}$ per realisation, RSR $=12$ dB, "
           r"$t_0=50$, 400 trials/point. Metric is eq. (20) in its Frobenius form.")
    d.image(FIGS / "b1_clean.png", height=0.215)
    d.caption("B1 — channel NMSE vs SNR at two pilot lengths. Both estimators "
              "operate on the exact model $\\mathbf{Z}=|\\mathbf{GS+B+W}|$; no "
              "linearisation is involved.")
    d.image(FIGS / "b2_clean.png", height=0.235)
    d.caption("B2 — channel NMSE vs pilot length at SNR $=5$ dB. $P=2K$ is the "
              "counting bound below which the problem is underdetermined.")
    d.para(
        "GS and EM-GS are nearly indistinguishable here because RSR $=12$ dB "
        "already drives $R(\\kappa)\\to1$. The measured separation grows as RSR "
        "falls: $+1.04$ dB at RSR $=0$, $+0.17$ dB at 12, $+0.01$ dB at 30.")

    d.new_page("7  Track B — structural estimation (cont.)")
    d.h2("7.2  The structural constraint")
    d.para(
        "Equation (5) says each channel column is a sum of $L_k$ complex "
        "exponentials. By Kronecker's theorem, a length-$N$ sequence is such a "
        "sum of $L$ exponentials if and only if its Hankel matrix has rank $L$. "
        "That gives an exact algebraic characterisation of the model — no grid, "
        "and the angles and gains never appear:")
    d.eq(r"\min_{\mathbf{G}}\ \bigl\|\,\mathbf{Z}-|\mathbf{GS}+\mathbf{B}|\,"
         r"\bigr\|_F^2 \quad \text{s.t.}\quad "
         r"\mathrm{rank}\,\mathcal{H}(\mathbf{g}_k)\le L_k", "(23)")
    d.para(
        "Three projections onto this set were implemented and compared. Cadzow "
        "was chosen because it is the only one that is simultaneously grid-free "
        "and a genuine alternating projection between two closed sets with "
        "exact projectors — SVD truncation (Eckart–Young) and anti-diagonal "
        "averaging onto the Hankel subspace. Angular OMP is grid-based and "
        "leaves about 6.9 % residual even on noiseless structured data; ESPRIT "
        "is grid-free but is a parameter estimator, not idempotent, with no "
        "variational characterisation.")
    d.h2("7.3  Why structure must be enforced inside the iteration")
    d.para(
        "Cui's row adapter is separable across receive elements, so an "
        "unstructured sweep cannot see a coupling along $n$. Applying the "
        "projection once and then running GS to convergence achieves nothing, "
        "because the fixed points of $T_{\\mathrm{GS}}^{\\infty}\\circ P_S$ are "
        "just the fixed points of $T_{\\mathrm{GS}}$. Measured: the gain decays "
        "monotonically with the number of unconstrained iterations after "
        "projection — $+1.30$ dB at 1, $+0.06$ at 10, exactly $0.00$ at 50. "
        "Interleaving instead gives the iteration map")
    d.eq(r"T \;=\; P_S \circ T_{\mathrm{GS}}", "(24)")
    d.para(
        "whose fixed points are simultaneously structured and consistent with "
        "the measurement update. The measurement step remains exactly eqs. "
        "(11)–(14) on $\\mathbf{Z}$.")
    d.h2("7.4  Identifiability — when structure can help")
    d.para(
        "A length-$N$ Hankel matrix has rank at most $\\lceil N/2\\rceil$. At "
        "$N=8$ that cap is 4, so for $L_k\\ge5$ — 60 % of the "
        "$\\mathcal{U}\\{3..7\\}$ prior — the true channel is already full rank "
        "and the constraint is vacuous. Counting real degrees of freedom, the "
        "unstructured model has $2NK$ and the geometric one $3\\sum_k L_k$:")
    d.table([
        [r"$N$", "unstructured $2NK$", r"geometric $3\sum L_k$", "reduction"],
        ["8", "48", "45", r"1.07$\times$"],
        ["16", "96", "45", r"2.13$\times$"],
        ["32", "192", "45", r"4.27$\times$"],
    ], [0.08, 0.24, 0.24, 0.18])
    d.para(
        "So at the frozen $N=8$ the prior carries almost no information, and "
        "measured gains are mixed ($+0.92$ dB at low SNR, $-1.48$ dB at high). "
        "At $N=16$ and $N=32$ the constraint becomes active and the gains turn "
        "consistently positive — up to $+3.86$ dB at $N=32$, where the "
        "structured estimator won every one of 40 trials. This is a property of "
        "the configuration, not of the algorithm.")


def _audit(d: Doc) -> None:
    d.new_page("8  Verification")
    d.para(
        "Every formula above was checked numerically by re-deriving it "
        "independently rather than reading the implementation and agreeing "
        "with it. 141 checks, no blocker, high or medium findings.")
    d.table([
        ["quantity", "independent reference", "result"],
        [r"$\sigma^2 = K/\mathrm{SNR}$", "derived from eq. (6)", "exact"],
        [r"$|\alpha_b|$", r"eq. (8); ruled out $\sqrt{K\cdot}$, $\sqrt{\cdot/K}$", "exact"],
        [r"$\mathbb{E}|w|^2$", "empirical over 400 worlds", r"ratio 1.000"],
        [r"$R(\kappa)$", "SciPy and quadrature of $I_n$", r"$<10^{-9}$"],
        ["Gray mapping", "walked all neighbours, $M=4,16,64$", "0 violations"],
        ["bit errors", "recounted from constellation table", "exact match"],
        ["CRLB / ZF gap", r"analytic $10\log_{10}2$", "3.0103 dB"],
        ["conjugation", "noiseless recovery of $\\mathbf{G}$", r"$2.4\times10^{-15}$"],
        ["modulation per figure", r"bits/trial $= K\log_2 M$", "confirmed"],
        ["linearisation leak", "runtime tripwire on HS-GS", "0 calls"],
        ["store integrity", "6 stores: fingerprint, duplicates", "0 duplicates"],
    ], [0.20, 0.36, 0.20])
    d.h2("8.1  Agreement with Cui")
    d.para(
        "Curves were extracted from the published PDF by colour segmentation "
        "with axis calibration, then compared both vertically (BER ratio) and "
        "horizontally (the SNR shift aligning the curves, which is the "
        "meaningful measure for a waterfall).")
    d.table([
        ["figure", "qualitative", "absolute", "discrepancy"],
        ["Fig. 5", "full", "offset", r"$\approx2$ dB"],
        ["Fig. 6", "full", "offset", r"$\approx2$ dB"],
        ["Fig. 7(a)", "full", "yes", r"$-0.90$ to $+0.05$ dB"],
        ["Fig. 7(b)", "full", "yes", r"$-0.10$ to $-0.15$ dB"],
        ["Fig. 8", "full", "yes", r"$-0.80$ to $-1.40$ dB"],
    ], [0.14, 0.16, 0.14, 0.30])
    d.para(
        "Two quantities Cui states in prose were reproduced without any curve "
        "extraction at all, which is the strongest available evidence since it "
        "bypasses the pixel reading entirely: the Fig. 7(b) EM-GS-to-ZF gap "
        "(stated 3–4 dB, measured 3.79–3.97) and the Fig. 8 improvement from "
        "RSR 0 to 20 dB (stated more than one order of magnitude, measured "
        "$39.1\\times$).")
    d.h2("8.2  The residual 2 dB in Figs. 5 and 6")
    d.para(
        "The offset is constant across both NMSE sweeps, which is the signature "
        "of channel statistics rather than of a solver, calibration or metric "
        "error — and the BER figures, which use the same channel, solvers and "
        "calibration, agree to within 0.1–1.4 dB. Two audits against the actual "
        "TR 38.901 specification found no zero-free-parameter correction that "
        "closes it. The cause is that Cui's own Table I fixes the angle, delay "
        "and cluster statistics that would otherwise pin the channel down, "
        "while leaving the array geometry and the orientation of "
        "$\\boldsymbol{\\mu}_{eg}$ unstated. A sweep showed the achievable "
        "shift spans 0.0–1.3 dB purely by rotating that unspecified frame, so "
        "the residual is not identifiable from the published information.")
    d.h2("8.3  What could not be reproduced")
    d.bullets([
        "CM-ZF — described only as extending a cited approximation to the "
        "biased phase-retrieval problem. That is not a specification, so it was "
        "deliberately not invented, and appears in no figure.",
        "The absolute NMSE level of Figs. 5–6, for the reason in Section 8.2.",
        "Fig. 8's intended modulation, which the paper itself states "
        "inconsistently; resolved empirically in Section 6.5.",
    ])


if __name__ == "__main__":
    build()
