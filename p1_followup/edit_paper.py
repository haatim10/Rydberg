from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'paper/paper1/full_revision/haatim_hsgs_revised.tex'
s=p.read_text()
marker=r'\section{Performance References}'
complexity=r'''\subsection{Computational Cost}
\label{sec:cost}

Let $R$ be the number of candidate ranks and $Q$ the number of Cadzow
sweeps per application. One EM-GS update costs $O(NKP)$ after the pilot
least-squares operator has been formed. A balanced Hankel matrix has both
dimensions proportional to $N$. Using a full SVD, one sweep costs
$O(KN^3)$ across the $K$ users; antidiagonal averaging costs $O(KN^2)$.
Selecting a small rank does not reduce this full-SVD cost.

With $T_{\rm sel}$ candidate updates and $T$ final updates, an upper
order bound for the iterative work is
\begin{equation}
O\!\left((RT_{\rm sel}+T)(NKP+QKN^3)\right).
\end{equation}
This uses $P$ as an upper bound on the fitting-set size. Full-rank
candidates bypass structural processing and cost less. The spectral
initializer adds $O(NPK^2+NK^3)$ work per fitting set. Forming the pilot
operator adds $O(PK^2+K^3)$. Our candidate fits share the same initializer
and pilot operator; both are recomputed for the all-pilot final fit.

The implementation batches candidates and users. Its iterative storage
scales as $O(R(NP+NK+KN^2)+PK+K^2)$, including the Hankel/SVD arrays.
The initializer additionally stores per-row diagnostics of order
$O(NPK+NK^2)$. Processing candidates or users sequentially can reduce
peak storage without changing the estimator. These expressions describe
scaling; measured times also depend on the implementation and hardware.

'''
assert marker in s;s=s.replace(marker,complexity+marker,1)
# Keep the concrete amplitude example, remove the second introduction to it.
start=s.index('A simple example can be used to understand $J$.')
end=s.index('This tells us the direction',start)
s=s[:start]+r'''For $s=x+jy$, the amplitude is $a=\sqrt{x^2+y^2}$. At $(x,y)=(3,4)$,
its gradient is
\begin{equation}
\nabla a=\begin{bmatrix}3/5 & 4/5\end{bmatrix}^{T}.
\end{equation}

'''+s[end:]
start=s.index('In practice, we use several pilot measurements.')
end=s.index(r'\subsection{Constrained',start)
s=s[:start]+'''Independent pilot measurements contribute information that adds to form
the full Fisher information matrix. Their directions depend on both the
pilots and the known reference. The spatial channel model provides
additional information, which motivates the constrained reference below.

'''+s[end:]
# Compact numerical illustration while keeping accessible explanation.
start=s.index('The value $\\rho$ can be understood')
end=s.index('Fig.~\\ref{fig:boundary}',start)
s=s[:start]+r'''The ratio $\rho$ describes how much of the available Hankel rank the
channel effectively uses. For example, $r_{\mathrm{eff}}=4$ and
$r_{\max}=16$ give $\rho=0.25$, whereas $r_{\mathrm{eff}}=14$ gives
$\rho=0.875$. The second channel leaves less room for a low-rank
approximation. This is an interpretation of the noiseless channel, not
a guarantee about the estimate.

'''+s[end:]
needle='Every comparison uses the same channel, pilots, reference and noise for'
pilot=r'''The pilots have equal expected power but are not forced to be orthogonal.
Their full row rank permits the EM least-squares solve; it does not by
itself guarantee identifiable magnitude-only recovery. Pilot phases,
conditioning and reference strength also affect the Fisher information.
Optimizing the pilots or allocating unequal user powers would change the
experimental design and is not studied here.

'''
s=s.replace(needle,pilot+needle,1)
needle='We use $N=32$, $K=3$, $P=30$, RSR $=12$~dB,\n$L\\in\\{3,7,14\\}$ paths per user'
why=r'''We choose this filter as a simple smooth alternative to hard truncation.
For $s_i\gg\tau$, it changes the component little; for $s_i\ll\tau$,
it suppresses the component strongly without an abrupt rank cutoff.
The RMS scale makes the threshold relative to the current matrix size in
amplitude. The zero factor supplies an exact unstructured option. This
choice tests whether soft suppression is useful; it is not derived as an
optimal denoiser for the correlated errors produced by Hankel lifting.

'''
assert needle in s;s=s.replace(needle,why+needle,1)
# Scope of richer structure without inventing common-user angles or time exponentials.
needle='The true effective rank remains an analysis quantity'
s=s.replace(needle,'''We process each user's spatial channel separately. Joint-user structure
would require a model for shared paths, which the present independent-angle
model does not provide. Likewise, the Gaussian pilot index is not assumed
to have the exponential structure used by the spatial Hankel lift.
Adaptive stopping and alternative parameter-selection rules remain open;
this study does not establish convergence of the combined updates.

'''+needle,1)
p.write_text(s)
