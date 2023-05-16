from pathlib import Path

import numpy as np
import perfplot

un = np.random.rand(3)


def for_loop(a):
    res = np.empty_like(a)
    for i in range(len(a)):
        res[i] = np.cross(un, a[i])
    return res


def cross(a):
    res = np.cross(un, a)
    return res


b = perfplot.bench(
    setup=lambda n: np.random.rand(n, 3),
    kernels=[
        for_loop,
        cross,
    ],
    labels=["for-loop", "cross"],
    n_range=[2**k for k in range(15)],
    xlabel="len(a)",
)

out = Path(__file__).with_suffix(".png")
out = out.parent / "plot" / out.name
# out.mkdir(parents=True, exist_ok=True)
b.save(
    out,
    logx=True,
    logy=True,
)
