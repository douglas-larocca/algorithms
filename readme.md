algorithms
===

```bash
python setup.py build_ext --inplace
python setup.py install
```

gcc `__builtin` and `libc` functions:

```python
>>> from algorithms import gcc_builtin, libc
>>> bits = '101010101011111010101111101010111110101011111010101000000000'
>>> n = int(bits, 2)
>>> gcc_builtin.ffsll(n) # find first set
10
>>> libc.ffsll(n) # find first set
10
```

```python
>>> from algorithms.numerical import bits
>>> n = int('11011010', 2)
>>> gcc_builtin.popcount(n)
5
>>> bits.popcount(n)
5
```

## Sorting and searching

```python
>>> from algorithms.searching import kth_smallest
... kth_smallest([3,3,4,3,5], 2)
```