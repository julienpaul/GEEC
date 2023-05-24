from pathlib import Path

import numpy as np
import perfplot


def add(a):
    _sum = 0.0
    for i in range(len(a)):
        _sum += a[i]
    return _sum


def add_assign(a):
    _sum = 0.0
    for i in range(len(a)):
        _sum = _sum + a[i]
    return _sum


b = perfplot.bench(
    setup=np.random.rand,  # lambda n: np.random.rand(n)
    kernels=[add, add_assign],
    labels=["add", "add_assign"],
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
