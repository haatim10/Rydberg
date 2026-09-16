"""Losslessly merge trial checkpoints into one portable file per cell."""
import numpy as np
from codex_verification.run import cells,OUT,source_hash
from codex_verification.analyze import load

def main():
    target=OUT/"trials";target.mkdir(exist_ok=True)
    for c in cells():
        data=load(c)
        path=target/(c["tag"]+".npz")
        np.savez_compressed(path,**data,fingerprint=source_hash())
        with np.load(path) as check:
            for key,val in data.items():np.testing.assert_array_equal(check[key],val)
    print(f"Consolidated {len(cells())} cells; every array verified losslessly.")

if __name__=="__main__":main()
