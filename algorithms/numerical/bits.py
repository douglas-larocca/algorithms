
def popcount(n):
    """
    http://graphics.stanford.edu/~seander/bithacks.html

        See Ian Ashdown's nice newsgroup post for more information on counting
        the number of bits set (also known as sideways addition). The best
        bit counting method was brought to my attention on October 5, 2005 by
        Andrew Shapira; he found it in pages 187-188 of Software Optimization
        Guide for AMD Athlon™ 64 and Opteron™ Processors. Charlie Gordon
        suggested a way to shave off one operation from the purely parallel
        version on December 14, 2005, and Don Clugston trimmed three more
        from it on December 30, 2005. I made a typo with Don's suggestion
        that Eric Cole spotted on January 8, 2006. Eric later suggested the
        arbitrary bit-width generalization to the best method on November 17,
        2006. On April 5, 2007, Al Williams observed that I had a line of
        dead code at the top of the first method.
    
    Ian Ashdown:
    
        We can operate on multiple fields within the input value
        simultaneously by first adding adjacent bits, then bit pairs, and so
        on. This process, which involves "binary magic numbers," takes V
        iterations in all cases, where V is the base-2 logarithm of N rounded
        to the next highest integer.

        Here's the function in C (with following explanations):

        ```c
        int FastSideSum( int value )
        {
          int i;
          int result = value;

          for (i = 0; i < V; i++)
            result = ((result >> S[i]) & B[i]) + (result & B[i]);

          return result;
        }
        ```

        Yes, there is still a loop involved. However, you can always unroll it
        for any reasonable value of V. Thus for V = 4 (i.e, N = 16):

        ```c
        int SuperFastSideSum_16( int value )
        {
          int result = value;

          result = ((result >> S[1]) & B[1]) + (result & B[1]);
          result = ((result >> S[2]) & B[2]) + (result & B[2]);
          result = ((result >> S[3]) & B[3]) + (result & B[3]);
          result = ((result >> S[4]) & B[4]) + (result & B[4]);

          return result;
        }
        ```

        If you look at the compiler output for these two functions and
        calculate the number of machine cycles required for each, it becomes
        evident that the binary magic number approach is preferable for large
        N's.

        Now, to explain `S[]` and `B[]`. The `B` array consists of binary magic
        numbers, which are thoroughly discussed in:

        Freed, Edwin E. 1983. "Binary Magic Numbers," Dr. Dobb's Journal
        Vol. 78 (April), pp. 24-37.

        Magic numbers in mathematics are numbers which have special or unusual
        properties when used in certain calculations. Freed examined a class
        of binary magic numbers that can be used to:

        1. Determine the positions of bits within words;
        2. Reverse, permute and map bits within words;
        3. Compute sideways (unweighted and weighted) sums and parity; and
        4. Convert Gray code values to their binary equivalents.

        Binary magic numbers offer improved (i.e., faster) algorithms over
        those that can otherwise be developed, typically by a factor of N /
        log-2(N).

        Freed's numbers are members of a sequence, where the Nth number of the
        sequence is itself an infinite sequence from right to left of 2**N 1's
        followed by 2**N 0's, followed 2**N 1's, and so on. The initial
        numbers are:

        ```
          ...0101010101010101
          ...0011001100110011
          ...0000111100001111
          ...0000000011111111
                          ...
        ```
        
        For a word size of 16 bits then, we have four "B-constants":
        
        ```
          B[1] = 0101010101010101
          B[2] = 0011001100110011
          B[3] = 0000111100001111
          B[4] = 0000000011111111
        ```
        
        There are three `B`-constants for words sizes (N) of 8 or fewer bits,
        five `B`-constants for words sizes of 17 to 32 bits, and so on. As
        mentioned above, N is the word size in bits, and V is the base-2
        logarithm of N rounded to the next highest integer. `B[1]` through `B[V]`
        are thus the binary magic numbers truncated to the N least-significant
        bits, and V is:
        
        ```
          N        V
          ----------
          8        3
          9-16     4
          17-32    5
          33-64    6
          ...
        ```
        
        The constants `S[1]` through `S[N]` are a table of the powers of 2. That
        is:
        
        ```
          S[1] = 2
          S[2] = 4
          S[3] = 8
          S[4] = 16
          ...
        ```
        
        Freed (who was at the Mathematics Department of the Harvey Mudd
        College in Claremont, California when he wrote his article)
        acknowledged the assistance of Donald Knuth, and noted that many of
        his algorithms were derived from the preprint version of Knuth's "The
        Art of Computer Programming - Combinatorial Algorithms (Volume 4)."
        Unless I missed it, however, this book was never published.

        (One paper of interest -- I wasn't aware of it myself until I began
        writing this response -- may be:

        Guibas, L., and J. Stolfi. 1981. "A Language for Bitmap
        Manipulation," ACM Transactions on Graphics 1(3):192-214

        which deals with transposing bit matrices. The paper likely refers to
        previous papers on the topic.)

        Ian Ashdown, P. Eng.
    """
    s = [1, 2, 4, 8, 16]
    b = [0x55555555, 0x33333333, 0x0F0F0F0F, 0x00FF00FF, 0x0000FFFF]
    n = n - ((n >> 1) & b[0])
    n = ((n >> s[1]) & b[1]) + (n & b[1])
    n = ((n >> s[2]) + n) & b[2]
    n = ((n >> s[3]) + n) & b[3]
    n = ((n >> s[4]) + n) & b[4]
    return n