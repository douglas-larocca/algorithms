import time
from functools import wraps

units = {'y': 1e-24,  # yocto
         'z': 1e-21,  # zepto
         'a': 1e-18,  # atto
         'f': 1e-15,  # femto
         'p': 1e-12,  # pico
         'n': 1e-9,   # nano
         'µ': 1e-6,   # micro
         'm': 1e-3,   # mili
         'c': 1e-2,   # centi
         'd': 1e-1,   # deci
         'k': 1e3,    # kilo
         'M': 1e6,    # mega
         'G': 1e9,    # giga
         'T': 1e12,   # tera
         'P': 1e15,   # peta
         'E': 1e18,   # exa
         'Z': 1e21,   # zetta
         'Y': 1e24,   # yotta
    }

def timer(number=1e5):
    assert number >= 1 and number <= 1e9
    def timer_(fn):
        """dynamically generate the timing function
        so no loops need to be run; for fine 
        precision this seems to make about 30% difference;
        this will be generated and inserted into
        the global namespace at the time the function
        is decorated.
        """
        exec('def __call_repeat(fn, *args, **kwargs):\n'+
             '    start = time.perf_counter()\n'+
             '    fn(*args, **kwargs)\n'*(number-1)+
             '    val = fn(*args, **kwargs)\n'+
             '    delta = time.perf_counter() - start\n'+
             '    dt = delta / {}\n'.format(number)+
             '    return delta, val, dt\n'+
             'globals().update({"__call_repeat": __call_repeat})')
        call_repeat = globals()['__call_repeat']
        @wraps(fn)
        def fn_(*args, **kwargs):
            delta, val, dt = call_repeat(fn, *args, **kwargs)
            strf = '{:2f} s'.format(dt)
            for unit, measure in sorted(units.items(), key=lambda x: -x[1]):
                if measure < dt:
                    strf = '{:.2f} {}s = {} s / {}'.format(
                        dt / measure, unit, delta, number)
                    break
            print(strf)
            return val
        return fn_
    return timer_