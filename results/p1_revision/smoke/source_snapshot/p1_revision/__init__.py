"""Isolated, reproducible experiments addressing the ThesisProj-7 review."""
import os

# Set before importing NumPy in command-line workers. Explicitly supplied
# values remain respected; scientific run commands set each of these to 1.
for _key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key,'1')
