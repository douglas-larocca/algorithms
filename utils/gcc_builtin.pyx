from numpy cimport *
cimport numpy as np
import numpy as np

cimport cython

import_array()

cdef float64_t FP_ERR = 1e-13

from libc.stdlib cimport malloc, free

from numpy cimport NPY_INT8 as NPY_int8
from numpy cimport NPY_INT16 as NPY_int16
from numpy cimport NPY_INT32 as NPY_int32
from numpy cimport NPY_INT64 as NPY_int64
from numpy cimport NPY_FLOAT16 as NPY_float16
from numpy cimport NPY_FLOAT32 as NPY_float32
from numpy cimport NPY_FLOAT64 as NPY_float64

from numpy cimport (int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t,
                    uint32_t, uint64_t, float32_t, float64_t)

int8 = np.dtype(np.int8)
int16 = np.dtype(np.int16)
int32 = np.dtype(np.int32)
int64 = np.dtype(np.int64)
float16 = np.dtype(np.float16)
float32 = np.dtype(np.float32)
float64 = np.dtype(np.float64)

cdef np.int8_t MINint8 = np.iinfo(np.int8).min
cdef np.int16_t MINint16 = np.iinfo(np.int16).min
cdef np.int32_t MINint32 = np.iinfo(np.int32).min
cdef np.int64_t MINint64 = np.iinfo(np.int64).min
cdef np.float32_t MINfloat32 = np.NINF
cdef np.float64_t MINfloat64 = np.NINF

cdef np.int8_t MAXint8 = np.iinfo(np.int8).max
cdef np.int16_t MAXint16 = np.iinfo(np.int16).max
cdef np.int32_t MAXint32 = np.iinfo(np.int32).max
cdef np.int64_t MAXint64 = np.iinfo(np.int64).max
cdef np.float32_t MAXfloat32 = np.inf
cdef np.float64_t MAXfloat64 = np.inf

cdef double NaN = <double> np.NaN
cdef double nan = NaN

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b

_return_false = lambda self, other: False
_return_true = lambda self, other: True

ctypedef fused numeric:
    int8_t
    int16_t
    int32_t
    int64_t

    uint8_t
    uint16_t
    uint32_t
    uint64_t

    float32_t
    float64_t
    
cdef extern int __builtin_ffs(int x)
cpdef ffs(int x):
    """Returns one plus the index of the least significant 1-bit of x, or if
    x is zero, returns zero.
    """
    return __builtin_ffs(x)

cdef extern int __builtin_clz(unsigned int x)
cpdef clz(unsigned int x):
    """Returns the number of leading 0-bits in x, starting at the most
    significant bit position. If x is 0, the result is undefined.
    """
    return __builtin_clz(x)

cdef extern int __builtin_ctz(unsigned int x)
cpdef ctz(unsigned int x):
    """Returns the number of trailing 0-bits in x, starting at the least
    significant bit position. If x is 0, the result is undefined.
    """
    return __builtin_ctz(x)

cdef extern int __builtin_clrsb(int x)
cpdef clrsb(int x):
    """Returns the number of leading redundant sign bits in x, i.e. the
    number of bits following the most significant bit that are identical to
    it. There are no special cases for 0 or other values.
    """
    return __builtin_clrsb(x)

cdef extern int __builtin_popcount(unsigned int x)
cpdef popcount(unsigned int x):
    """Returns the number of 1-bits in x.
    """
    return __builtin_popcount(x)

cdef extern int __builtin_parity(unsigned int x)
cpdef parity(unsigned int x):
    """Returns the parity of x, i.e. the number of 1-bits in x modulo 2.
    """
    return __builtin_parity(x)

cdef extern int __builtin_ffsl(long x)
cpdef ffsl(long x):
    """Similar to __builtin_ffs, except the argument type is long.
    """
    return __builtin_ffsl(x)

cdef extern int __builtin_clzl(unsigned long x)
cpdef clzl(unsigned long x):
    """Similar to __builtin_clz, except the argument type is unsigned long.
    """
    return __builtin_clzl(x)

cdef extern int __builtin_ctzl(unsigned long x)
cpdef ctzl(unsigned long x):
    """Similar to __builtin_ctz, except the argument type is unsigned long.
    """
    return __builtin_ctzl(x)

cdef extern int __builtin_clrsbl(long x)
cpdef clrsbl(long x):
    """Similar to __builtin_clrsb, except the argument type is long.
    """
    return __builtin_clrsbl(x)

cdef extern int __builtin_popcountl(unsigned long x)
cpdef popcountl(unsigned long x):
    """Similar to __builtin_popcount, except the argument type is unsigned
    long.
    """
    return __builtin_popcountl(x)

cdef extern int __builtin_parityl(unsigned long x)
cpdef parityl(unsigned long x):
    """Similar to __builtin_parity, except the argument type is unsigned long.
    """
    return __builtin_parityl(x)

cdef extern int __builtin_ffsll(long long x)
cpdef ffsll(long long x):
    """Similar to __builtin_ffs, except the argument type is long long.
    """
    return __builtin_ffsll(x)

cdef extern int __builtin_clzll(unsigned long long x)
cpdef clzll(unsigned long long x):
    """Similar to __builtin_clz, except the argument type is unsigned long
    long.
    """
    return __builtin_clzll(x)

cdef extern int __builtin_ctzll(unsigned long long x)
cpdef ctzll(unsigned long long x):
    """Similar to __builtin_ctz, except the argument type is unsigned long
    long.
    """
    return __builtin_ctzll(x)

cdef extern int __builtin_clrsbll(long long x)
cpdef clrsbll(long long x):
    """Similar to __builtin_clrsb, except the argument type is long long.
    """
    return __builtin_clrsbll(x)

cdef extern int __builtin_popcountll(unsigned long long x)
cpdef popcountll(unsigned long long x):
    """Similar to __builtin_popcount, except the argument type is unsigned
    long long.
    """
    return __builtin_popcountll(x)

cdef extern int __builtin_parityll(unsigned long long x)
cpdef parityll(unsigned long long x):
    """Similar to __builtin_parity, except the argument type is unsigned long
    long.
    """
    return __builtin_parityll(x)

cdef extern double __builtin_powi(double x, int y)
cpdef powi(double x, int y):
    """Returns the first argument raised to the power of the second. Unlike
    the pow function no guarantees about precision and rounding are made.
    """
    return __builtin_powi(x, y)

cdef extern float __builtin_powif(float x, int y)
cpdef powif(float x, int y):
    """Similar to __builtin_powi, except the argument and return types are
    float.
    """
    return __builtin_powif(x, y)

cdef extern long double __builtin_powil(long double x, int y)
cpdef powil(long double x, int y):
    """Similar to __builtin_powi, except the argument and return types are
    long double.
    """
    return __builtin_powil(x, y)

cdef extern uint16_t __builtin_bswap16(uint16_t x)
cpdef bswap16(uint16_t x):
    """Returns x with the order of the bytes reversed; for example, 0xaabb
    becomes 0xbbaa. Byte here always means exactly 8 bits.
    """
    return __builtin_bswap16(x)

cdef extern uint32_t __builtin_bswap32(uint32_t x)
cpdef bswap32(uint32_t x):
    """Similar to __builtin_bswap16, except the argument and return types are
    32 bit.
    """
    return __builtin_bswap32(x)

cdef extern uint64_t __builtin_bswap64(uint64_t x)
cpdef bswap64(uint64_t x):
    """Similar to __builtin_bswap32, except the argument and return types are
    64 bit.
    """
    return __builtin_bswap64(x)

'''
cdef extern int __builtin_types_compatible_p(type1, type2)
cpdef types_compatible_p(type1, type2):
    """You can use the built-in function __builtin_types_compatible_p to
    determine whether two types are the same.
    
    This built-in function returns 1 if the unqualified versions of the types
    type1 and type2 (which are types, not expressions) are compatible, 0
    otherwise. The result of this built-in function can be used in integer
    constant expressions.
    
    This built-in function ignores top level qualifiers (e.g., const,
    volatile). For example, int is equivalent to const int.
    
    The type int[] and int[5] are compatible. On the other hand, int and char
    * are not compatible, even if the size of their types, on the particular
    architecture are the same. Also, the amount of pointer indirection is
    taken into account when determining similarity. Consequently, short * is
    not similar to short **. Furthermore, two types that are typedefed are
    considered compatible if their underlying types are compatible.
    
    An enum type is not considered to be compatible with another enum type
    even if both are compatible with the same integer type; this is what the
    C standard specifies. For example, enum {foo, bar} is not similar to
    enum {hot, dog}.
    
    You typically use this function in code whose execution varies depending
    on the arguments' types. For example:
    
    #define foo(x)
    ({
    typeof (x) tmp = (x);
    if (__builtin_types_compatible_p (typeof (x), long
    double))                 tmp = foo_long_double (tmp);
    else if (__builtin_types_compatible_p (typeof
    (x), double))                 tmp = foo_double (tmp);
    else if (__builtin_types_compatible_p
    (typeof (x), float))                  tmp = foo_float (tmp);
    else
    abort ();
    tmp;
    })
    Note: This construct is only available for C.
    """
    return __builtin_types_compatible_p(type1, type2)

cdef extern type __builtin_call_with_static_chain(call_exp, pointer_exp)
cpdef call_with_static_chain(call_exp, pointer_exp):
    """The call_exp expression must be a function call, and the pointer_exp
    expression must be a pointer. The pointer_exp is passed to the function
    call in the target's static chain location. The result of builtin is the
    result of the function call.
    
    Note: This builtin is only available for C. This builtin can be used to
    call Go closures from C.
    """
    return __builtin_call_with_static_chain(call_exp, pointer_exp)

cdef extern type __builtin_choose_expr(const_exp, exp1, exp2)
cpdef choose_expr(const_exp, exp1, exp2):
    """You can use the built-in function __builtin_choose_expr to evaluate
    code depending on the value of a constant expression. This built-in
    function returns exp1 if const_exp, which is an integer constant
    expression, is nonzero. Otherwise it returns exp2.
    
    This built-in function is analogous to the ‘? :’ operator in C, except
    that the expression returned has its type unaltered by promotion rules.
    Also, the built-in function does not evaluate the expression that is not
    chosen. For example, if const_exp evaluates to true, exp2 is not
    evaluated even if it has side-effects.
    
    This built-in function can return an lvalue if the chosen argument is an
    lvalue.
    
    If exp1 is returned, the return type is the same as exp1's type.
    Similarly, if exp2 is returned, its return type is the same as exp2.
    
    Example:
    
    #define foo(x)
    __builtin_choose_expr (
    __builtin_types_compatible_p (typeof (x), double),
    foo_double (x),
    __builtin_choose_expr (
    __builtin_types_compatible_p (typeof (x),
    float),                           foo_float (x),
    /* The void expression results in a
    compile-time error                     when assigning the result to
    something.  */                          (void)0))
    Note: This construct is only available for C. Furthermore, the unused
    expression (exp1 or exp2 depending on the value of const_exp) may still
    generate syntax errors. This may change in future revisions.
    """
    return __builtin_choose_expr(const_exp, exp1, exp2)

cdef extern type __builtin_complex(real, imag)
cpdef complex(real, imag):
    """The built-in function __builtin_complex is provided for use in
    implementing the ISO C11 macros CMPLXF, CMPLX and CMPLXL. real and imag
    must have the same type, a real binary floating-point type, and the
    result has the corresponding complex type with real and imaginary parts
    real and imag. Unlike ‘real + I * imag’, this works even when
    infinities, NaNs and negative zeros are involved.
    """
    return __builtin_complex(real, imag)

cdef extern int __builtin_constant_p(exp)
cpdef constant_p(exp):
    """You can use the built-in function __builtin_constant_p to determine if
    a value is known to be constant at compile time and hence that GCC can
    perform constant-folding on expressions involving that value. The
    argument of the function is the value to test. The function returns the
    integer 1 if the argument is known to be a compile-time constant and 0
    if it is not known to be a compile-time constant. A return of 0 does not
    indicate that the value is not a constant, but merely that GCC cannot
    prove it is a constant with the specified value of the -O option.
    
    You typically use this function in an embedded application where memory
    is a critical resource. If you have some complex calculation, you may
    want it to be folded if it involves constants, but need to call a
    function if it does not. For example:
    
    #define Scale_Value(X)                  (__builtin_constant_p
    (X)             ? ((X) * SCALE + OFFSET) : Scale (X))
    You may use this built-in function in either a macro or an inline
    function. However, if you use it in an inlined function and pass an
    argument of the function as the argument to the built-in, GCC never
    returns 1 when you call the inline function with a string constant or
    compound literal (see Compound Literals) and does not return 1 when you
    pass a constant numeric value to the inline function unless you specify
    the -O option.
    
    You may also use __builtin_constant_p in initializers for static data.
    For instance, you can write
    
              static const int table[] = {
                 __builtin_constant_p (EXPRESSION) ? (EXPRESSION) : -1,
                 /* ... */
              };
    This is an acceptable initializer even if EXPRESSION is not a constant
    expression, including the case where __builtin_constant_p returns 1
    because EXPRESSION can be folded to a constant but EXPRESSION contains
    operands that are not otherwise permitted in a static initializer (for
    example, 0 && foo ()). GCC must be more conservative about evaluating
    the built-in in this case, because it has no opportunity to perform
    optimization.
    """
    return __builtin_constant_p(exp)

cdef extern long __builtin_expect(long exp, long c)
cpdef expect(long exp, long c):
    """You may use __builtin_expect to provide the compiler with branch
    prediction information. In general, you should prefer to use actual
    profile feedback for this (-fprofile-arcs), as programmers are
    notoriously bad at predicting how their programs actually perform.
    However, there are applications in which this data is hard to collect.
    
    The return value is the value of exp, which should be an integral
    expression. The semantics of the built-in are that it is expected that
    exp == c. For example:
    
              if (__builtin_expect (x, 0))
                foo ();
    indicates that we do not expect to call foo, since we expect x to be
    zero. Since you are limited to integral expressions for exp, you should
    use constructions such as
    
              if (__builtin_expect (ptr != NULL, 1))
                foo (*ptr);
    when testing pointer or floating-point values.
    """
    return __builtin_expect(exp, c)

cdef extern void __builtin_trap(void)
cpdef trap(void):
    """This function causes the program to exit abnormally. GCC implements
    this function by using a target-dependent mechanism (such as
    intentionally executing an illegal instruction) or by calling abort. The
    mechanism used may vary from release to release so you should not rely
    on any particular implementation.
    """
    return __builtin_trap(void)

cdef extern void __builtin_unreachable(void)
cpdef unreachable(void):
    """If control flow reaches the point of the __builtin_unreachable, the
    program is undefined. It is useful in situations where the compiler
    cannot deduce the unreachability of the code.
    
    One such case is immediately following an asm statement that either never
    terminates, or one that transfers control elsewhere and never returns.
    In this example, without the __builtin_unreachable, GCC issues a warning
    that control reaches the end of a non-void function. It also generates
    code to return after the asm.
    
              int f (int c, int v)
              {
                if (c)
                  {
                    return v;
                  }
                else
                  {
                    asm("jmp error_handler");
                    __builtin_unreachable ();
                  }
              }
    Because the asm statement unconditionally transfers control out of the
    function, control never reaches the end of the function body. The
    __builtin_unreachable is in fact unreachable and communicates this fact
    to the compiler.
    
    Another use for __builtin_unreachable is following a call a function that
    never returns but that is not declared __attribute__((noreturn)), as in
    this example:
    
              void function_that_never_returns (void);
    
              int g (int c)
              {
                if (c)
                  {
                    return 1;
                  }
                else
                  {
                    function_that_never_returns ();
                    __builtin_unreachable ();
                  }
              }
    """
    return __builtin_unreachable(void)

cdef extern void *__builtin_assume_aligned(const void *exp, size_t align, ...)
cpdef assume_aligned(const void *exp, size_t align, ...):
    """This function returns its first argument, and allows the compiler to
    assume that the returned pointer is at least align bytes aligned. This
    built-in can have either two or three arguments, if it has three, the
    third argument should have integer type, and if it is nonzero means
    misalignment offset. For example:
    
              void *x = __builtin_assume_aligned (arg, 16);
    means that the compiler can assume x, set to arg, is at least 16-byte
    aligned, while:
    
              void *x = __builtin_assume_aligned (arg, 32, 8);
    means that the compiler can assume for x, set to arg, that (char *) x - 8
    is 32-byte aligned.
    """
    return __builtin_assume_aligned(*exp, align, ...)

cdef extern int __builtin_LINE()
cpdef LINE():
    """This function is the equivalent to the preprocessor __LINE__ macro and
    returns the line number of the invocation of the built-in. In a C++
    default argument for a function F, it gets the line number of the call
    to F.
    """
    return __builtin_LINE()

cdef extern const char * __builtin_FUNCTION()
cpdef FUNCTION():
    """This function is the equivalent to the preprocessor __FUNCTION__ macro
    and returns the function name the invocation of the built-in is in.
    """
    return __builtin_FUNCTION()

cdef extern const char * __builtin_FILE()
cpdef FILE():
    """This function is the equivalent to the preprocessor __FILE__ macro and
    returns the file name the invocation of the built-in is in. In a C++
    default argument for a function F, it gets the file name of the call to
    F.
    """
    return __builtin_FILE()

cdef extern void __builtin___clear_cache(char *begin, char *end)
cpdef __clear_cache(char *begin, char *end):
    """This function is used to flush the processor's instruction cache for
    the region of memory between begin inclusive and end exclusive. Some
    targets require that the instruction cache be flushed, after modifying
    memory containing code, in order to obtain deterministic behavior.
    
    If the target does not require instruction cache flushes,
    __builtin___clear_cache has no effect. Otherwise either instructions are
    emitted in-line to clear the instruction cache or a call to the
    __clear_cache function in libgcc is made.
    """
    return __builtin___clear_cache(*begin, *end)

cdef extern void __builtin_prefetch(const void *addr, ...)
cpdef prefetch(const void *addr, ...):
    """This function is used to minimize cache-miss latency by moving data
    into a cache before it is accessed. You can insert calls to
    __builtin_prefetch into code for which you know addresses of data in
    memory that is likely to be accessed soon. If the target supports them,
    data prefetch instructions are generated. If the prefetch is done early
    enough before the access then the data will be in the cache by the time
    it is accessed.
    
    The value of addr is the address of the memory to prefetch. There are two
    optional arguments, rw and locality. The value of rw is a compile-time
    constant one or zero; one means that the prefetch is preparing for a
    write to the memory address and zero, the default, means that the
    prefetch is preparing for a read. The value locality must be a
    compile-time constant integer between zero and three. A value of zero
    means that the data has no temporal locality, so it need not be left in
    the cache after the access. A value of three means that the data has a
    high degree of temporal locality and should be left in all levels of
    cache possible. Values of one and two mean, respectively, a low or
    moderate degree of temporal locality. The default is three.
    
              for (i = 0; i < n; i++)
                {
                  a[i] = a[i] + b[i];
                  __builtin_prefetch (&a[i+j], 1, 1);
                  __builtin_prefetch (&b[i+j], 0, 1);
                  /* ... */
                }
    Data prefetch does not generate faults if addr is invalid, but the
    address expression itself must be valid. For example, a prefetch of
    p->next does not fault if p->next is not a valid address, but evaluation
    faults if p is not a valid address.
    
    If the target does not support data prefetch, the address expression is
    evaluated if it includes side effects but no other code is generated and
    GCC does not issue a warning.
    """
    return __builtin_prefetch(*addr, ...)

cdef extern double __builtin_huge_val(void)
cpdef huge_val(void):
    """Returns a positive infinity, if supported by the floating-point
    format, else DBL_MAX. This function is suitable for implementing the ISO
    C macro HUGE_VAL.
    """
    return __builtin_huge_val(void)

cdef extern float __builtin_huge_valf(void)
cpdef huge_valf(void):
    """Similar to __builtin_huge_val, except the return type is float.
    """
    return __builtin_huge_valf(void)

cdef extern long double __builtin_huge_vall(void)
cpdef huge_vall(void):
    """Similar to __builtin_huge_val, except the return type is long double.
    """
    return __builtin_huge_vall(void)

cdef extern int __builtin_fpclassify(int, int, int, int, int, ...)
cpdef fpclassify(int, int, int, int, int, ...):
    """This built-in implements the C99 fpclassify functionality. The first
    five int arguments should be the target library's notion of the possible
    FP classes and are used for return values. They must be constant values
    and they must appear in this order: FP_NAN, FP_INFINITE, FP_NORMAL,
    FP_SUBNORMAL and FP_ZERO. The ellipsis is for exactly one floating-point
    value to classify. GCC treats the last argument as type-generic, which
    means it does not do default promotion from float to double.
    """
    return __builtin_fpclassify(int, int, int, int, int, ...)

cdef extern double __builtin_inf(void)
cpdef inf(void):
    """Similar to __builtin_huge_val, except a warning is generated if the
    target floating-point format does not support infinities.
    """
    return __builtin_inf(void)

cdef extern _Decimal32 __builtin_infd32(void)
cpdef infd32(void):
    """Similar to __builtin_inf, except the return type is _Decimal32.
    """
    return __builtin_infd32(void)

cdef extern _Decimal64 __builtin_infd64(void)
cpdef infd64(void):
    """Similar to __builtin_inf, except the return type is _Decimal64.
    """
    return __builtin_infd64(void)

cdef extern _Decimal128 __builtin_infd128(void)
cpdef infd128(void):
    """Similar to __builtin_inf, except the return type is _Decimal128.
    """
    return __builtin_infd128(void)

cdef extern float __builtin_inff(void)
cpdef inff(void):
    """Similar to __builtin_inf, except the return type is float. This
    function is suitable for implementing the ISO C99 macro INFINITY.
    """
    return __builtin_inff(void)

cdef extern long double __builtin_infl(void)
cpdef infl(void):
    """Similar to __builtin_inf, except the return type is long double.
    """
    return __builtin_infl(void)

cdef extern int __builtin_isinf_sign(...)
cpdef isinf_sign(...):
    """Similar to isinf, except the return value is -1 for an argument of
    -Inf and 1 for an argument of +Inf. Note while the parameter list is an
    ellipsis, this function only accepts exactly one floating-point
    argument. GCC treats this parameter as type-generic, which means it does
    not do default promotion from float to double.
    """
    return __builtin_isinf_sign(...)

cdef extern double __builtin_nan(const char *str)
cpdef nan(const char *str):
    """This is an implementation of the ISO C99 function nan.
    
    Since ISO C99 defines this function in terms of strtod, which we do not
    implement, a description of the parsing is in order. The string is
    parsed as by strtol; that is, the base is recognized by leading ‘0’ or
    ‘0x’ prefixes. The number parsed is placed in the significand such that
    the least significant bit of the number is at the least significant bit
    of the significand. The number is truncated to fit the significand field
    provided. The significand is forced to be a quiet NaN.
    
    This function, if given a string literal all of which would have been
    consumed by strtol, is evaluated early enough that it is considered a
    compile-time constant.
    """
    return __builtin_nan(*str)

cdef extern _Decimal32 __builtin_nand32(const char *str)
cpdef nand32(const char *str):
    """Similar to __builtin_nan, except the return type is _Decimal32.
    """
    return __builtin_nand32(*str)

cdef extern _Decimal64 __builtin_nand64(const char *str)
cpdef nand64(const char *str):
    """Similar to __builtin_nan, except the return type is _Decimal64.
    """
    return __builtin_nand64(*str)

cdef extern _Decimal128 __builtin_nand128(const char *str)
cpdef nand128(const char *str):
    """Similar to __builtin_nan, except the return type is _Decimal128.
    """
    return __builtin_nand128(*str)

cdef extern float __builtin_nanf(const char *str)
cpdef nanf(const char *str):
    """Similar to __builtin_nan, except the return type is float.
    """
    return __builtin_nanf(*str)

cdef extern long double __builtin_nanl(const char *str)
cpdef nanl(const char *str):
    """Similar to __builtin_nan, except the return type is long double.
    """
    return __builtin_nanl(*str)

cdef extern double __builtin_nans(const char *str)
cpdef nans(const char *str):
    """Similar to __builtin_nan, except the significand is forced to be a
    signaling NaN. The nans function is proposed by WG14 N965.
    """
    return __builtin_nans(*str)

cdef extern float __builtin_nansf(const char *str)
cpdef nansf(const char *str):
    """Similar to __builtin_nans, except the return type is float.
    """
    return __builtin_nansf(*str)

cdef extern long double __builtin_nansl(const char *str)
cpdef nansl(const char *str):
    """Similar to __builtin_nans, except the return type is long double.
    """
    return __builtin_nansl(*str)
'''