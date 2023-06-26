import timeit

setup = """
import numpy as np
from itertools import combinations, pairwise
face=np.array([3,2,1])
def f1(face):
    b = np.append(np.flip(face), face[-1])
    return [p for p in pairwise(b)]

def f2(face):
    b = np.flip(face)
    return [(x, y) for x, y in zip(b, np.roll(b, -1))]

def f3(face):
    return [p for p in combinations(face,2)]

"""
timeit.timeit("f3(face)", setup=setup)

setup = """
import numpy as np
arr=np.random.rand(3)
def f1(arr):
    np.linalg.norm(arr)

def f2(arr):
    np.sqrt(np.sum([i**2 for i in arr]))
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
arr=np.random.rand(3)
def f1(arr):
    np.linalg.norm(arr)

def f2(arr):
    np.sqrt(np.sum(arr*arr))
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
arr=[np.random.rand(3),np.random.rand(3)]
def f1(arr):
    np.linalg.norm(np.array(arr), axis=1)

def f2(arr):
    np.linalg.norm(arr, axis=1)

def f3(arr):
    np.linalg.norm(arr[0])
    np.linalg.norm(arr[1])

def f4(arr):
    np.linalg.norm([arr[0],arr[1]], axis=1)
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
import vg
arr=np.random.rand(3)
def f1(arr):
    n1 = arr / np.linalg.norm(arr)

def f2(arr):
    n1 = vg.normalize(arr)
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
import vg
arr=np.random.rand(2,3)
def f1(arr):
    n1 = arr.T / np.linalg.norm(arr, axis=1)

def f2(arr):
    n1 = vg.normalize(arr)
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
arr=np.random.rand(2,3)
def f1(arr):
    return np.cross(arr[0],arr[1])

def f2(arr):
    if len(arr[0]) == 3 and len(arr[1]) == 3:
        a1, a2, a3 = arr[0]
        b1, b2, b3 = arr[1]

        x = a2 * b3 - a3 * b2
        y = a3 * b1 - a1 * b3
        z = a1 * b2 - a2 * b1
        return np.array([x, y, z])
    else:
        raise TypeError("Input vectors must be 3D")
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
arr=np.random.rand(2,3)
def f1(arr):
    return np.dot(arr[0],arr[1])

def f2(arr):
    if len(arr[0]) == 3 and len(arr[1]) == 3:
        a1, a2, a3 = arr[0]
        b1, b2, b3 = arr[1]

        x = a1*b1+a2*b2+a3*b3
        return x
    else:
        raise TypeError("Input vectors must be 3D")
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
a=np.random.rand(10)
def f1(a):
    for x in a:
        y=x**2

def f2(a):
    for x in a:
        y=x*x
"""
timeit.timeit("f1(a)", setup=setup)

setup = """
import numpy as np
a=np.random.rand(2)
a1, a2 = a
shift = 1.
def f1(a, shift):
    y = a - shift
def f2(a1, a2, shift):
    y1, y2 = a1 - shift, a2 - shift
"""
timeit.timeit("f1(a)", setup=setup)

setup = """
import numpy as np
a=np.random.rand(2,3)
a1, a2 = a
shift = 1.
def f1(a):
    y1, y2 = np.linalg.norm(a, axis=1)
def f2(a1, a2):
    y1, y2 = np.linalg.norm(a1), np.linalg.norm(a2)
"""
timeit.timeit("f1(a)", setup=setup)

setup = """
import numpy as np
L,b=np.random.rand(2)
def f1(b, L):
    b2 = 0.5 * b / L
def f2(b, L):
    b2 = b / (2 * L)
def f3(b, L):
    b2 = b / 2 / L
def f4(b, L):
    b2 = b / L / 2
"""
timeit.timeit("f1(b, L)", setup=setup, number=10000000)

setup = """
import numpy as np
L,b=np.random.rand(2)
def f1(b, L):
    b2 = (1 / L) * b
def f2(b, L):
    b2 = b / L
"""
timeit.timeit("f1(b, L)", setup=setup, number=10000000)

setup = """
from typer.testing import CliRunner
from geec.cli import app

def f1():
    runner = CliRunner()
    result = runner.invoke(app, "test")
def f2():
    runner = CliRunner()
    result = runner.invoke(app, "test-grad")
"""
# timeit.timeit("f1()", setup=setup, number=100)
print("%.2f usec/pass" % (1000 * timeit.timeit("f1()", setup=setup, number=100) / 100))

setup = """
import numpy as np
arr=np.random.rand(3)
def f1(arr):
    n1 = arr / 2.5

def f2(arr):
    n1 = np.zeros(3)
def f3(arr):
    n1 = np.zeros((3,3))
def f4(arr):
    n1 = 0
def f5(arr):
    n1 = None
"""
timeit.timeit("f1(arr)", setup=setup)

setup = """
import numpy as np
arr=np.random.rand(3)
def f1(arr):
    n1 = len(arr)
def f2(arr):
    n1 = arr.size
"""
timeit.timeit("f1(arr)", setup=setup, number=10000000)

setup = """
import numpy as np
def f1():
    n1 = np.diag(np.full(3, -1))
def f2():
    n1 = - np.identity(3)
"""
timeit.timeit("f1()", setup=setup)

setup = """
import numpy as np
from geec.utils import cross_product
arr=np.random.rand(3,3)
def f1(arr):
    cross = [cross_product(arr[i], arr[(i + 1) % 3]) for i in range(3)]

def f1b(arr):
    shift = np.roll(arr,-1)
    cross = [cross_product(arr[i], shift[i]) for i in range(3)]

def f2(arr):
    cross = [np.cross(arr[i], arr[(i + 1) % 3]) for i in range(3)]

def f3(arr):
    shift = np.roll(arr,-1)
    cross = list(map(cross_product,arr,shift))

def f4(arr):
    shift = np.roll(arr,-1)
    cross = list(map(np.cross,arr,shift))
"""
timeit.timeit("f1(arr)", setup=setup)
