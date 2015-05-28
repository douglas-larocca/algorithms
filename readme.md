algorithms
===

```bash
python setup.py build_ext --inplace
python setup.py install
```

gcc `__builtin` and `libc` functions:

```python
>>> from algorithms import gcc_builtin
>>> from algorithms.utils import libc
>>> bits = '101010101011111010101111101010111110101011111010101000000000'
>>> n = int(bits, 2)
>>> gcc_builtin.ffsll(n) # find first set
10
>>> libc.ffsll(n) # find first set
10
```

## timer decorator

```python
>>> from algorithms.utils import timer
>>> from algorithms.numerical import bits
>>> gcc_popcount = timer(10000)(gcc_builtin.popcount)
>>> python_popcount = timer(10000)(bits.popcount)
>>> gcc_popcount(int('1111011101100101', 2))
77.06 ns = 0.0007706498727202415 s / 10000
11
>>> python_popcount(int('1111011101100101', 2))
2.55 µs = 0.02548007108271122 s / 10000
11
```

## Sorting and searching

```python
>>> from algorithms.searching import kth_smallest
>>> kth_smallest([3,3,4,3,5], 2)
3
```