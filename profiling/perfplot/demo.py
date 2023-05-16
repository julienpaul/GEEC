from pathlib import Path

import numpy as np
import perfplot


def use_append(size):
    out = []
    for i in range(size):
        out.append(i)
    return out


def list_compr(size):
    return [i for i in range(size)]  # noqa: C416


def list_range(size):
    return list(range(size))


# perfplot.show(
b = perfplot.bench(
    setup=lambda n: n,
    kernels=[
        use_append,
        list_compr,
        list_range,
        np.arange,
        lambda n: list(np.arange(n)),
    ],
    labels=["use_append", "list_compr", "list_range", "numpy", "list_numpy"],
    n_range=[2**k for k in range(15)],
    xlabel="len(a)",
    equality_check=None,
)

out = Path(__file__).with_suffix(".png")
out = out.parent / "plot" / out.name
b.save(out)
