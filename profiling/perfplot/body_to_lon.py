from operator import methodcaller
from pathlib import Path

import perfplot

import geec.config
import geec.mass

config = geec.config._default_setup()
masses = geec.mass.get_masses(config)


def f0(n):
    arr = masses * n
    list(map(lambda x: x.to_lon180(), arr))  # noqa: C417


def f1(n):
    arr = masses * n
    [mass.to_lon180() for mass in arr]


def f2(n):
    arr = masses * n
    list(map(methodcaller("to_lon180"), arr))


b = perfplot.bench(
    setup=lambda n: n,
    kernels=[f0, f1, f2],
    labels=["map", "comprehension", "methodcaller"],
    n_range=list(range(10)),
    xlabel="len(a)",
    equality_check=None,  # set to None to disable "correctness" assertion
)
out = Path(__file__).with_suffix(".png")
outdir = out.parent / "plot"
outdir.mkdir(parents=True, exist_ok=True)
out = outdir / out.name

b.save(out)
