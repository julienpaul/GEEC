from pathlib import Path

import numpy as np
import pandas as pd

import perfplot

# applying a function to a list


def square(x):
    return x**2


def for_list(a):
    y = 0.0
    for _x in a:
        y = square(a)
    return y


def np_apply_along(a):
    y = np.apply_along_axis(square, axis=0, arr=a)
    return y


def comprehension(a):
    y = [square(x) for x in a]
    return y


def pd_apply(a):
    df = pd.DataFrame(a, columns=["x"])
    y = df.apply(square, axis=1, result_type="expand")
    return y


b = perfplot.bench(
    setup=lambda n: np.random.rand(n),  # or setup=np.random.rand
    kernels=[
        for_list,
        np_apply_along,
        comprehension,
        pd_apply,
    ],
    labels=["for_list", "np_apply_along", "comprehension", "pd_apply"],
    n_range=[2**k for k in range(20)],
    xlabel="len(a)",
    # More optional arguments with their default values:
    # logx="auto",  # set to True or False to force scaling
    # logy="auto",
    equality_check=None,  # np.allclose,  # set to None to disable "correctness" assertion
    # show_progress=True,
    # target_time_per_measurement=1.0,
    # max_time=None,  # maximum time per measurement
    # time_unit="s",  # set to one of ("auto", "s", "ms", "us", or "ns") to force plot units
    # relative_to=1,  # plot the timings relative to one of the measurements
    # flops=lambda n: 3*n,  # FLOPS plots
)

out = Path(__file__).with_suffix(".png")
out = out.parent / "plot" / out.name
b.save(
    out,
    logx=True,
    logy=True,
)
