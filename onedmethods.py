import time

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def arg_check(func, beg, end, eps):
    if func is None:
        raise AttributeError("Function is None")
    if beg is None:
        raise AttributeError("Left edge of area is None")
    if end is None:
        raise AttributeError("Right edge of area is None")
    if eps is None:
        raise AttributeError("Epsilon is None")
    if beg > end:
        raise AttributeError("Wrong area")
    if eps == 0:
        raise AttributeError("Epsilon is 0")
    #return True


def passive_search(func, beg, end, eps):
    arg_check(func, beg, end, eps)
    tmin = beg
    fmin = func(tmin)
    range_list = frange(beg, end, eps)
    n = 0
    for t in range_list:
        cur = func(t)
        n += 1
        if cur <= fmin:
            fmin = cur
            tmin = t
        else:
            break
    return [tmin, n]


#  https://en.wikipedia.org/wiki/Bisection_method
def dichotomi_search(func, beg, end, eps):
    arg_check(func, beg, end, eps)
    delta = eps / 2
    left, right = beg, end
    n = 0 # function calculate counter
    while right - left > 2*eps:
        c = (right + left - delta) / 2.0
        d = (right + left + delta) / 2.0
        if func(c) < func(d):
            right = d
        else:
            left = c
        n += 2
    tmin = (left + right) / 2.0
    return [tmin, n]


#  https://en.wikipedia.org/wiki/Golden_section_search
def gold_section_method(func, beg, end, eps):
    arg_check(func, beg, end, eps)
    if end - beg < 2*eps:
        return (end + beg) / 2.0
    left, right = beg, end
    c = left + (right - left) * (3 - 5**(.5)) / 2.0
    d = right - (right - left) * (3 - 5**(.5)) / 2.0
    fc, fd = func(c), func(d)
    n = 2 # function calculate counter
    while right - left > 2*eps:
        if fc < fd:
            right = d
            if right - left < 2*eps:
                break
            d, fd = c, fc
            c = left + (right - left) * (3 - 5**(.5)) / 2.0
            fc = func(c)
        else:
            left = c
            if right - left < 2*eps:
                break
            c, fc = d, fd
            d = right - (right - left) * (3 - 5**(.5)) / 2.0
            fd = func(d)
        n += 1
    return [(right + left) / 2.0, n]


def fib(n):
    a, b = 0, 1
    for i in range(n):
        temp = b
        b = a + b
        a = temp
    return b


def fibbonachi_method(func, beg, end, eps):
    arg_check(func, beg, end, eps)
    m = 0
    while fib(m + 2) < (end - beg) / eps:
        m += 1
    left, right = beg, end
    c = left + (right - left)*(fib(m)/fib(m + 2))
    d = left + (right - left)*(fib(m + 1)/fib(m + 2))
    fc, fd = func(c), func(d)
    n = 2 # function calculate counter
    for i in range(m):
        if fc < fd:
            right = d
            d, fd = c, fc
            c = left + (right - left)*(fib(m - i)/fib(m - i + 2))
            fc = func(c)
        else:
            left = c
            c, fc = d, fd
            d = left + (right - left)*(fib(m - i + 1)/fib(m - i + 2))
            fd = func(d)
        n += 1
    return [(right + left) / 2.0, n]


def tangents_search(func, grad, beg, end, eps):
    arg_check(func, beg, end, eps)
    count = 0
    if end - beg < 2 * eps:
        return [(end + beg) / 2.0, count]

    left = beg
    right = end
    atemp = grad(left)*left - func(left)
    btemp = func(right) - grad(right)*right
    ctemp = grad(left) - grad(right)
    mid = (atemp + btemp) / (ctemp)
    while grad(mid) > eps and (right - left) > 2 * eps:
        if grad(mid) < 0:
            left = mid
        else:
            right = mid
        a = grad(left)*left - func(left)
        b = func(right) - func(right)*right
        c = grad(left) - grad(right)
        count += 1
        mid = (a + b) / (c)
    return [mid, count]


def nuton_raffson(func, grad, gradd, beg, end, eps):
    lastp = beg
    point = lastp - grad(lastp) / gradd(lastp)
    count = 1
    while abs(lastp - point) > 2*eps and abs(grad(point)) > eps:
        lastp = point
        point = lastp - grad(lastp) / gradd(lastp)
        count += 1
    return [point, count]


def tangents_method(func, grad, beg, end, eps):
    count = 1
    lastlastp = beg
    lastp = end
    point = lastp - grad(lastp) * (lastlastp - lastp) / (grad(lastlastp) - grad(lastp))
    while abs(lastp - point) > 2 * eps and abs(grad(point)) > eps:
        lastlastp = lastp
        lastp = point
        point = lastp - grad(lastp) * (lastlastp - lastp) / (grad(lastlastp) - grad(lastp))
        count += 1
    return [point, count]


# F(X)= 3x^4 - 10x^3 + 21x^2 + 12x
def fun1746(x):
    f = 3*x**4 - 10*x**3 + 21*x**2 + 12*x
    return f


def fun1746_g(x):
    return 12*x**3 - 30*x**2 + 42*x + 12


def fun1746_gg(x):
    return 36*x**2 - 60*x**1 + 42


def sqr_x(x):
    f = x**2
    return f


def sqr_d(x):
    return 2*x


def sqr_dd(x):
    return 2


def run_all_methods(file, func, grad, gradd, beg, end, eps):
    print("## ", func.__name__, "  \non [", beg, ",", end, "]", end="  \n\n---  \n\n", file=file)

    start = time.time()
    ans = passive_search(func, beg, end, eps)
    print("__Passive search__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

    start = time.time()
    ans = dichotomi_search(func, beg, end, eps)
    print("__Dihotomy search__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

    start = time.time()
    ans = gold_section_method(func, beg, end, eps)
    print("__Gold section search__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

    start = time.time()
    ans = fibbonachi_method(func, beg, end, eps)
    print("__Fibbonachi method__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

    start = time.time()
    ans = tangents_search(func, grad, beg, end, eps)
    print("__Tangents__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

    start = time.time()
    ans = nuton_raffson(func, grad, gradd, beg, end, eps)
    print("__Nuton-Raffson__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

    start = time.time()
    ans = tangents_method(func, grad, beg, end, eps)
    print("__Tangents(chords)__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)

if __name__ == '__main__':
    file = open('onedmethods.md','w')
    run_all_methods(file, fun1746, fun1746_g, fun1746_gg, 0, 0.5, 0.01)
    run_all_methods(file, fun1746, fun1746_g, fun1746_gg, -2, 4, 0.01)
    run_all_methods(file, sqr_x, sqr_d, sqr_dd, -5, 5, 0.01)