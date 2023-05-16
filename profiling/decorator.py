"""Adapted from  https://www.youtube.com/watch?v=8qEnExGLZfY"""
import cProfile
import io
import pstats
from pathlib import Path


def profile(fnc):
    """A decorator that uses cProfile to profile a function"""
    dirout = Path(__file__)
    dirout = dirout.parent / "prof"
    dirout.mkdir(parents=True, exist_ok=True)
    out = dirout / fnc.__name__
    out = out.with_suffix(".prof")

    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        ps.dump_stats(out)
        print(s.getvalue())
        return retval

    return inner
