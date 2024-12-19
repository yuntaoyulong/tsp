from numba import jit
import random
@jit(nopython=True)
def random_selection(rate):
    
    rate_new = [int((10**5) * round(r, 5)) for r in rate]
    randnum = random.randint(1, sum(rate_new))
    start = 0
    for index, scope in enumerate(rate_new):
        start += scope
        if randnum <= start:
            return index