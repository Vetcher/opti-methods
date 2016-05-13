import time


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


class OptiFunc(object):
    def __init__(self, func, grad=None, params=None, infunc=None, ingrad=None, name=None):
        self.par = params
        self.func = func
        self.gradient = grad
        self.infunc = infunc
        self.ingrad = ingrad
        self.name = name

    def value(self, x):
        if self.par is None:
            return self.func(x)
        if self.infunc is None:
            return self.func(x, self.par)
        return self.func(x, self.par, self.infunc)

    def grad(self, x):
        if self.gradient is None:
            return 0
        if self.par is None:
            return self.gradient(x)
        if self.ingrad is None:
            return self.gradient(x, self.par)
        return self.gradient(x, self.par, self.ingrad)


def passive_search(ofunc, gradd, beg, end, eps):
    iter = 0
    tmin = beg
    fmin = ofunc.value(tmin)
    range_list = frange(beg, end, eps)
    n = 0
    for t in range_list:
        iter += 1
        cur = ofunc.value(t)
        n += 1
        if cur <= fmin:
            fmin = cur
            tmin = t
        else:
            break
    return [tmin, n, iter]


#  https://en.wikipedia.org/wiki/Bisection_method
def dichotomi_search(ofunc, gradd, beg, end, eps):
    delta = eps / 2
    left, right = beg, end
    n = 0  # function calculate counter
    iter = 0
    while right - left > 2*eps:
        c = (right + left - delta) / 2.0
        d = (right + left + delta) / 2.0
        if ofunc.value(c) < ofunc.value(d):
            right = d
        else:
            left = c
        n += 2
        iter += 1
    tmin = (left + right) / 2.0
    return [tmin, n, iter]


#  https://en.wikipedia.org/wiki/Golden_section_search
def gold_section_method(ofunc, gradd, beg, end, eps):
    if end - beg < 2*eps:
        return (end + beg) / 2.0
    left, right = beg, end
    c = left + (right - left) * (3 - 5**.5) / 2.0
    d = right - (right - left) * (3 - 5**.5) / 2.0
    fc, fd = ofunc.value(c), ofunc.value(d)
    n = 2  # function calculate counter
    iter = 0
    while right - left > 2*eps:
        if fc < fd:
            right = d
            if right - left < 2*eps:
                break
            d, fd = c, fc
            c = left + (right - left) * (3 - 5**(.5)) / 2.0
            fc = ofunc.value(c)
        else:
            left = c
            if right - left < 2*eps:
                break
            c, fc = d, fd
            d = right - (right - left) * (3 - 5**(.5)) / 2.0
            fd = ofunc.value(d)
        n += 1
        iter += 1
    return [(right + left) / 2.0, n, iter]


def fib(n):
    a, b = 0, 1
    for i in range(n):
        temp = b
        b += a
        a = temp
    return b


def fibbonachi_method(ofunc, gradd, beg, end, eps):
    m = 0
    while fib(m + 2) < (end - beg) / eps:
        m += 1
    left, right = beg, end
    c = left + (right - left)*(fib(m)/fib(m + 2))
    d = left + (right - left)*(fib(m + 1)/fib(m + 2))
    fc, fd = ofunc.value(c), ofunc.value(d)
    n = 2 # function calculate counter
    iter = 0
    for i in range(m):
        if fc < fd:
            right = d
            d, fd = c, fc
            c = left + (right - left)*(fib(m - i)/fib(m - i + 2))
            fc = ofunc.value(c)
        else:
            left = c
            c, fc = d, fd
            d = left + (right - left)*(fib(m - i + 1)/fib(m - i + 2))
            fd = ofunc.value(d)
        n += 1
        iter += 1
    return [(right + left) / 2.0, n, iter]


def tangents_search(ofunc, gradd, beg, end, eps):
    count = 0
    if end - beg < 2 * eps:
        return [(end + beg) / 2.0, count]

    left = beg
    right = end
    atemp = ofunc.grad(left)*left - ofunc.value(left)
    btemp = ofunc.value(right) - ofunc.grad(right)*right
    ctemp = ofunc.grad(left) - ofunc.grad(right)
    mid = (atemp + btemp) / ctemp
    iter = 0
    while ofunc.grad(mid) > eps and (right - left) > 2 * eps:
        if ofunc.grad(mid) < 0:
            left = mid
        else:
            right = mid
        a = ofunc.grad(left)*left - ofunc.value(left)
        b = ofunc.value(right) - ofunc.value(right)*right
        c = ofunc.grad(left) - ofunc.grad(right)
        count += 1
        mid = (a + b) / c
        iter += 1
    return [mid, count, iter]


def nuton_raffson(ofunc, gradd, beg, end, eps):
    lastp = beg
    point = lastp - ofunc.grad(lastp) / gradd(lastp)
    count = 1
    iter = 0
    while abs(lastp - point) > 2*eps and abs(ofunc.grad(point)) > eps:
        lastp = point
        point = lastp - ofunc.grad(lastp) / gradd(lastp)
        count += 1
        iter += 1
    return [point, count, iter]


def tangents_method(ofunc, gradd, beg, end, eps):
    count = 1
    lastlastp = beg
    lastp = end
    point = lastp - ofunc.grad(lastp) * (lastlastp - lastp) / (ofunc.grad(lastlastp) - ofunc.grad(lastp))
    iter = 0
    while abs(lastp - point) > 2 * eps and abs(ofunc.grad(point)) > eps:
        lastlastp = lastp
        lastp = point
        point = lastp - ofunc.grad(lastp) * (lastlastp - lastp) / (ofunc.grad(lastlastp) - ofunc.grad(lastp))
        count += 1
        iter += 1
    return [point, count, iter]


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


allmethods = [
    passive_search,
    dichotomi_search,
    gold_section_method,
    fibbonachi_method,
    tangents_search,
    tangents_method,
    nuton_raffson
]

methods_names = {
    'passive_search': 'Метод пассивного поиска',
    'dichotomi_search': 'Метод дихотомии',
    'gold_section_method': 'Метод золотого сечения',
    'fibbonachi_method': 'Метод через последовательность Фиббоначи',
    'tangents_search': 'Метод касательных',
    'tangents_method': 'Метод хорд/секущих',
    'nuton_raffson': 'Метод Ньютона-Раффсона'
}


def run_method(output, method, ofunc, gradd, beg, end, eps):
    print("#### Метод: ", methods_names.get(method.__name__), file=output)
    start = time.time()
    ans = method(ofunc, gradd, beg, end, eps)
    print("__Ответ__:  ", file=output)
    print("`Xmin` =", ans[0], end="  \n", file=file)
    print("`F(Xmin)` = ", ofunc.value(ans[0]), end="  \n", file=file)
    print("__Производительность__:  ", file=output)
    print("`Вызовов функций` =", ans[1], end="  \n", file=file)
    print("`Итерации` =", ans[2], end="  \n", file=output)
    print("`Время` = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)


def run_all_methods(output, ofunc, gradd, beg, end, eps):
    print("## ", ofunc.name, "  \n#####Отрезок [", beg, ",", end, "]", "`\u03B5`=", eps, end="  \n\n---  \n\n", file=file)
    for each in allmethods:
        run_method(output, each, ofunc, gradd, beg, end, eps)


if __name__ == '__main__':
    file = open('onedmethods.md','w', encoding='utf-8')
    ofunc = OptiFunc(fun1746, fun1746_g, name='Номер 17.46')
    run_all_methods(file, ofunc, fun1746_gg, 0, 0.5, 0.01)
    run_all_methods(file, ofunc, fun1746_gg, -2, 4, 0.01)
    ofunc = OptiFunc(sqr_x, sqr_d, name='F(x)= x*x')
    run_all_methods(file, ofunc, sqr_dd, -5, 5, 0.01)
