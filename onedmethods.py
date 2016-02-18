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


def passive_search(file, func, beg, end, eps):
    print("\n-----------------------------------\n", file=file)
    print("\n-----------------------------------\n", file=file)
    arg_check(func, beg, end, eps)
    print("##__Passive search__\n _a_=", beg, "_b_=", end, "_eps_= ", eps, end='  \n', file=file)
    tmin = beg
    fmin = func(tmin)
    range_list = frange(beg, end, eps)
    n = 0
    print("\n-----------------------------------\n", file=file)
    for t in range_list:
        cur = func(t)
        print("\t_t_=", t, "F(t)=", cur, end='  \n', file=file)
        n += 1
        if cur <= fmin:
            fmin = cur
            tmin = t
        else:
            break
    print("\n-----------------------------------\n", file=file)
    print("Function was calculated", n, "times.", end='  \n', file=file)
    return tmin

#  https://en.wikipedia.org/wiki/Bisection_method
def dichotomi_search(file, func, beg, end, eps):
    print("\n-----------------------------------\n", file=file)
    print("\n-----------------------------------\n", file=file)
    arg_check(func, beg, end, eps)
    print("##__Dihotomy search__\n _a_=", beg, "_b_=", end, "_eps_= ", eps, end='  \n', file=file)
    delta = eps / 2
    left, right = beg, end
    n = 0 # function calculate counter
    print("\n-----------------------------------\n", file=file)
    while right - left > 2*eps:
        c = (right + left - delta) / 2.0
        d = (right + left + delta) / 2.0
        print("\t_a_=", left, "_c_=", c, "_d_=", d, "_b_=", right, end='  \n', file=file)
        if func(c) < func(d):
            right = d
        else:
            left = c
        n += 2
    tmin = (left + right) / 2.0
    print("\n-----------------------------------\n", file=file)
    print("Function was calculated", n, "times.", end='  \n', file=file)
    return tmin

#  https://en.wikipedia.org/wiki/Golden_section_search
def gold_section_method(file, func, beg, end, eps):
    print("\n-----------------------------------\n", file=file)
    print("\n-----------------------------------\n", file=file)
    arg_check(func, beg, end, eps)
    print("##__Gold section search__\n _a_=", beg, "_b_=", end, "_eps_= ", eps, end='  \n', file=file)
    if end - beg < 2*eps:
        return (end + beg) / 2.0
    left, right = beg, end
    c = left + (right - left) * (3 - 5**(.5)) / 2.0
    d = right - (right - left) * (3 - 5**(.5)) / 2.0
    fc, fd = func(c), func(d)
    n = 2 # function calculate counter
    print("\n-----------------------------------\n", file=file)
    while right - left > 2*eps:
        print("\t_a_=", left, "_c_=", c, "_d_=", d, "_b_=", right, end='  \n', file=file)
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
    print("\n-----------------------------------\n", file=file)
    print("Function was calculated", n, "times.", end='  \n', file=file)
    return (right + left) / 2.0


def fib(n):
    a, b = 0, 1
    for i in range(n):
        temp = b
        b = a + b
        a = temp
    return b


def fibbonachi_method(file, func, beg, end, eps):
    print("\n-----------------------------------\n", file=file)
    print("\n-----------------------------------\n", file=file)
    arg_check(func, beg, end, eps)
    print("##__Fibbonachi search__\n _a_=", beg, "_b_=", end, "_eps_= ", eps, end='  \n', file=file)
    m = 0
    while fib(m + 2) < (end - beg) / eps:
        m += 1
    print("n=", m, file=file)
    left, right = beg, end
    c = left + (right - left)*(fib(m)/fib(m + 2))
    d = left + (right - left)*(fib(m + 1)/fib(m + 2))
    fc, fd = func(c), func(d)
    n = 2 # function calculate counter
    print("\n-----------------------------------\n", file=file)
    for i in range(m):
        print("\t_a_=", left, "_c_=", c, "_d_=", d, "_b_=", right, end='  \n', file=file)
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
    print("\n-----------------------------------\n", file=file)
    print("Function was calculated", n, "times.", end='  \n', file=file)
    return (right + left) / 2.0

def fun1746(x):
    f = 3*x**4 - 10*x**3 + 21*x**2 + 12*x
    return f


def sqr_x(x):
    f = x**2
    return f

def run_all_methods(file, func, beg, end, eps):
    Passive_search = passive_search(file, func, beg, end, eps)
    print("_Xmin_ =", Passive_search, end=' ', file=file)
    print("_F(Xmin)_ = ", func(Passive_search), file=file)

    Dicho_search = dichotomi_search(file, func, beg, end, eps)
    print("_Xmin_ =", Dicho_search, end=' ', file=file)
    print("_F(Xmin)_ = ", func(Dicho_search), file=file)

    Gold_search = gold_section_method(file, func, beg, end, eps)
    print("_Xmin_ =", Gold_search, end=' ', file=file)
    print("_F(Xmin)_ = ", func(Gold_search), file=file)

    Fib_search = fibbonachi_method(file, func, beg, end, eps)
    print("_Xmin_ =", Fib_search, end=' ', file=file)
    print("_F(Xmin)_ = ", func(Fib_search), file=file)

if __name__ == '__main__':
    file = open('readme.md','w')
    run_all_methods(file, fun1746, 0, 0.5, 0.01)
    #run_all_methods(sqr_x, -5, 5, 0.01)