import math
from pathlib import Path

import numpy as np
import perfplot


def for_sum(a):
    _sum = 0.0
    for i in range(len(a)):
        _sum += a[i]
    return _sum


def for_sum2(a):
    _sum = 0.0
    for i in a:
        _sum += i
    return _sum


b = perfplot.bench(
    setup=np.random.rand,  # lambda n: np.random.rand(n)
    kernels=[for_sum, for_sum2, np.sum, sum, math.fsum],
    labels=["for-sum", "for_sum2", "numpy.sum", "sum", "math.fsum"],
    n_range=[2**k for k in range(18)],
    xlabel="len(a)",
)

out = Path(__file__).with_suffix(".png")
out = out.parent / "plot" / out.name
b.save(
    out,
    logx=True,
    logy=True,
)
