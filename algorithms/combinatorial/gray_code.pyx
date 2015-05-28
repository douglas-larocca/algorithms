"""gray codes

def composition_gray_code(n):
    '''
    gray code for integer compositions

    [[6, 0, 0, 0, 0, 0],
     [1, 1, 4, 0, 0, 0],
     [1, 1, 1, 3, 0, 0],
     [1, 1, 1, 1, 2, 0],
     ...
     [3, 1, 1, 1, 0, 0],
     [4, 1, 1, 0, 0, 0],
     [5, 1, 0, 0, 0, 0],
     [5, 1, 0, 0, 0, 0]]

    This implements Algorithm 3 of [1], with the only
    modification that the line `a[pos + 2] -= 1`
    must be added to properly zero-pad the right side
    of the arrays.

    [1] Toufik Mansour and Ghalib Nassar. Loop-Free Gray Code 
    Algorithms for the Set of Compositions. Journal of 
    Mathematical Modelling and Algorithms 9(4) 343-356. 
    2010-12-01. 10.1007/s10852-010-9131-3
    '''
    a = [0]*(n+1)
    yield [n]+[0]*(n-1)
    a[1] = 1
    a[2] = n - 1
    pos = 1
    yield a[1:]
    while pos > 0:
        if a[pos + 1] > 1:
            pos += 1
            a[pos + 1] = a[pos] - 1
            a[pos] = 1
            yield a[1:]
        else:
            pos -= 1
            if pos > 0: 
                a[pos] += 1
                a[pos + 2] -= 1
                yield a[1:]
"""