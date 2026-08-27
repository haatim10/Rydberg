# Track D - verification gate results

Rendered from `reports/trackD_verify.json`. Do not edit by hand.

| Gate | Pass | Summary |
|---|---|---|
| `A_shapes` | PASS | 4 shape configs + 5 negative cases |
| `B_forward_float64` | PASS | max rel 9.799e-17 < 1e-12 |
| `C_ls_parity_float64` | PASS | max rel 5.597e-13 < 1e-12 |
| `D_gs_degeneration_float64` | PASS | rel 5.871e-13 < 1e-12 |
| `E_emgs_degeneration_float64` | PASS | rel 5.700e-13 < 1e-10; bessel vs scipy 2.488e-16 |
| `B_forward_float32` | PASS | max rel 5.678e-08 < 1e-05 |
| `C_ls_parity_float32` | PASS | max rel 1.479e-07 < 1e-05 |
| `D_gs_degeneration_float32` | PASS | rel 1.910e-07 < 1e-05 |
| `E_emgs_degeneration_float32` | PASS | rel 2.429e-07 < 1e-05; bessel vs scipy 2.488e-16 |
| `F_transformer_identity` | PASS | max abs diff 0.0e+00 (required exactly 0.0) |
| `G_conjugation` | PASS | max rel 9.748e-17; sign e^-jnpsi; conj cross-checks 1.41/1.41 (large) |
| `H_gradients` | PASS | loss 0.2243; all grads finite and > 0 across 3 layers |
| `J_noiseless_fixed_point` | PASS | fixed point 3.28e-14, GS limit 3.06e-14 |
| `K_kappa_invariance` | PASS | max rel 1.612e-16; negative control 0.75 (must be large) |
| `I_overfit32` | PASS | best -136.12 dB (target < -25.0 dB) |

**15/15 gates pass.**

## Model

- FilterNet: 970
- Gates: 10
- Transformer: 1,585,920
- **Total: 1,586,900**
- Initial alphas: [0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088, 0.11920291930437088]

