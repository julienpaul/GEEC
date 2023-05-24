from pathlib import Path

import numpy as np
import perfplot

val = np.nan


def fill(n):
    a = np.empty(n)
    a.fill(val)
    return a


def colon(n):
    a = np.empty(n)
    a[:] = val
    return a


def full(n):
    return np.full(n, val)


def ones_times(n):
    return val * np.ones(n)


def nplist(n):
    return np.array(n * [val])


b = perfplot.bench(
    setup=lambda n: n,
    kernels=[fill, colon, full, ones_times, nplist],
    n_range=[2**k for k in range(20)],
    xlabel="len(a)",
    equality_check=None,  # set to None to disable "correctness" assertion
)
out = Path(__file__).with_suffix(".png")
outdir = out.parent / "plot"
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / out.name

b.save(out)
