from pathlib import Path

import numpy as np
import perfplot


def np_linalg_norm(a):
    y = np.linalg.norm(a, axis=1)
    return y


def comprehension(a):
    y = [np.linalg.norm(x) for x in a]
    return y


# compare different methods
b = perfplot.bench(
    setup=lambda n: np.random.rand(n, 3),  # or setup=np.random.rand
    kernels=[
        np_linalg_norm,
        comprehension,
    ],
    labels=["np_linalg_norm", "comprehension"],
    n_range=[2**k for k in range(15)],
    xlabel="len(a)",
    # More optional arguments with their default values:
    # logx="auto",  # set to True or False to force scaling
    # logy="auto",
    # equality_check=np.allclose,  # set to None to disable "correctness" assertion
    # show_progress=True,
    # target_time_per_measurement=1.0,
    # max_time=None,  # maximum time per measurement
    # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
    # relative_to=1,  # plot the timings relative to one of the measurements
    # flops=lambda n: 3*n,  # FLOPS plots
)

out = Path(__file__).with_suffix(".png")
out = out.parent / "plot" / out.name
b.save(out)
