# Submission checks: results

Fresh paired tests. Positive gain means lower error for the named method. Read both dataset banks.

| Bank | Method | Reference | Mean gain (dB) | Conditional adjusted interval | Worst SNR gain |
|---|---|---|---:|---|---:|
| bank1 | matched/adaptive | matched/supervised | 1.015 | [0.9746994994799266, 1.0566051600145707] | -0.008 |
| bank1 | matched/adaptive | matched/final_only | 0.697 | [0.6628835349869787, 0.7338802780349466] | 0.258 |
| bank1 | matched/gate_one | matched/adaptive | 0.064 | [0.060461916224237196, 0.06804632520518432] | -0.035 |
| bank1 | rep1/adaptive | rep1/global | 2.252 | [2.1938260983861455, 2.307303914214559] | 0.461 |
| bank1 | rep1/adaptive | rep1/none | 4.250 | [4.173263822032976, 4.3243828831075275] | 3.262 |
| bank1 | rep1/adaptive | hs_gs | 1.939 | [1.8323752155349613, 2.037449218249452] | 0.786 |
| bank2 | matched/adaptive | matched/supervised | 1.012 | [0.9742509705796203, 1.049017275168492] | -0.011 |
| bank2 | matched/adaptive | matched/final_only | 0.694 | [0.6616733351041815, 0.7280007393711535] | 0.271 |
| bank2 | matched/gate_one | matched/adaptive | 0.066 | [0.062435346161217375, 0.06978113153709736] | -0.021 |
| bank2 | rep2/adaptive | rep2/global | 2.185 | [2.127473980823445, 2.2416594171148905] | 0.258 |
| bank2 | rep2/adaptive | rep2/none | 4.191 | [4.120197199691019, 4.259725029244497] | 3.472 |
| bank2 | rep2/adaptive | hs_gs | 1.837 | [1.732349912742258, 1.9399312598806848] | 0.809 |

All per-cell, severe-error, selected/final and supervised-selection sensitivity results are in summary.json.
Weak-reference and two-path stress cases are reported separately from the primary SNR average.
These intervals capture test-world variation conditional on trained weights, not uncertainty across dataset populations.
Do not claim equivalence from a nonsignificant contrast. Review audit.json and timing.json before manuscript use.
