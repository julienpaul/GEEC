from pathlib import Path

import numpy as np
import perfplot


def diag_one(x):
    for _a in x:
        A = np.diag(-np.ones(3), 0)
    return A


def diag_full(x):
    for _a in x:
        A = np.diag(np.full(3, -1))
    return A


def zero_slice(x):
    for _a in x:
        A = np.zeros((3, 3), int)
        A[range(3), range(3)] = -1
    return A


# P1=np.tile(p1, (3, 1))
# np.cross(A,P1)

# C.dot(n1)

b = perfplot.bench(
    setup=lambda n: np.random.rand(n),  # or setup=np.random.rand
    kernels=[
        diag_one,
        diag_full,
        zero_slice,
    ],
    labels=["diag_one", "diag_full", "zero_slice"],
    n_range=[2**k for k in range(20)],
    xlabel="len(a)",
    # More optional arguments with their default values:
    # logx="auto",  # set to True or False to force scaling
    # logy="auto",
    # equality_check=None,  # set to None to disable "correctness" assertion
    # show_progress=True,
    # target_time_per_measurement=1.0,
    # max_time=None,  # maximum time per measurement
    # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns")
    #                 # to force plot units
    # relative_to=1,  # plot the timings relative to one of the measurements
    # flops=lambda n: 3*n,  # FLOPS plots
)

out = Path(__file__).with_suffix(".png")
outdir = out.parent / "plot"
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / out.name
b.save(
    out,
    logx=True,
    logy=True,
)
