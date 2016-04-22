import time
from matrix import *
import numpy


# F(X)= sqrt(1 + x[0]^2 + x[1]^2)
def f17102(x):
    return (1 + x[0]**2 + x[1]**2)**.5


def f17102_g(x):
    return [x[0] / (1 + x[0]**2 + x[1]**2)**.5, x[1] / (1 + x[0]**2 + x[1]**2)**.5]


def f17102_dg(x):
    a11 = (1+x[0]**2+x[1]**2)**.5 * ((1+x[0]**2+x[1]**2)**2 - x[1]**2)
    a22 = (1+x[0]**2+x[1]**2)**.5 * ((1+x[0]**2+x[1]**2)**2 - x[0]**2)
    a12 = x[0]*x[1]*(1+x[0]**2+x[1]**2)**.5
    ans = [[a11, a12], [a12, a22]]
    return ans


def fsqr(x):
    return (x[0]-5)**2 + (x[1]-3)**4


def fsqrt_g(x):
    return [2*(x[0] - 5), 4*(x[1]-3)**3]


def fsqrt_dg(x):
    a22 = 1.0 / (12*(x[1]-3)**2)
    a11 = 1.0 / 2
    a12 = 0
    ans = [[a11, a12], [a12, a22]]
    return ans


def gold_section_in_space(func, x, eps, dir):
    cur = list(x)
    count = 1
    f_prev = func(cur)
    prev = list(cur)
    direction = list(dir)
    delta = 0.1  # first step

    old = list(prev)
    for i in range(len(prev)):         # make small step to start search
        prev[i] -= delta*direction[i]  # delta for use gold section method
    count += 1

    f_for_delta = func(prev)  # func value for new point

    while f_for_delta <= f_prev:
        delta = delta * 1.3 + delta**.5  # variate this function
        for i in range(len(prev)):  # do new step
            prev[i] = old[i] - delta*direction[i]
        count += 1
        f_for_delta = func(prev)

    # Golden section method
    right = delta
    left = 0
    left_p = old
    right_p = prev

    # Copy left probe
    c_arg = list(left_p)
    c = left + (right - left) * (3 - 5**.5) / 2.0
    for i in range(len(c_arg)):
        c_arg[i] -= c * direction[i]

    # Copy right probe
    d_arg = list(right_p)
    d = right - (right - left) * (3 - 5**.5) / 2.0
    for i in range(len(d_arg)):
        d_arg[i] += d * direction[i]

    fc, fd = func(c_arg), func(d_arg)  # func values in probe points
    count += 2
    while abs(right - left) > 2*eps:  # gold section loop
        if fc < fd:
            right = d
            if abs(right - left) < 2*eps:
                break
            d, fd = c, fc
            c = left + (right - left) * (3 - 5**.5) / 2.0
            c_arg = list(left_p)
            for i in range(len(c_arg)):
                c_arg[i] -= c * direction[i]
            count += 1
            fc = func(c_arg)
        else:
            left = c
            if abs(right - left) < 2*eps:
                break
            c, fc = d, fd
            d_arg = list(right_p)
            d = right - (right - left) * (3 - 5**.5) / 2.0
            for i in range(len(d_arg)):
                d_arg[i] += d * direction[i]
            count += 1
            fd = func(d_arg)
    delta = (right + left) / 2.0
    for i in range(len(cur)):
        cur[i] = old[i] - delta * direction[i]

    return cur, count


# ||X|| = sqrt(sum(x[i]**2))
def norma(x):
    ans = 0
    for i in range(len(x)):
        ans += x[i]**2
    return ans**0.5


def alf_function(a, params):
    return params[0](params[1] - a * params[2])


# params[0] current x, [1]: grad(x), [2]: direction
def coordinate_alf(a, params, func):
    copy = list(params[0])
    copy[params[2]] = params[0][params[2]] - a * params[1][params[2]]
    return func(copy)


def coordinate_alf_g(a, params, grad):
    copy = list(params[0])
    copy[params[2]] = params[0][params[2]] - a * params[1][params[2]]
    return -params[1][params[2]]*grad(copy)


def alf_function_g(a, params):
    return -params[2]*params[0](params[1] - a*params[2])


# Градиентный спуск с фиксированным шагом
def gradient_fix_step(func, grad, x, eps):
    cur = list(x)
    alf = 0.1
    count = 0
    # Do - While loop
    while True:
        for i in range(len(x)):
            cur[i] -= alf * grad(cur)[i]
        count += 1
        if norma(grad(cur)) < eps:  # loop escape
            break
    return [cur, count]


# Градиентный спуск с дроблением шага
def gradient_change_step(func, grad, x, eps):
    cur = list(x)
    lbd = 0.5
    count = 0
    # Do - While loop
    while True:
        alf = 10
        next = [0, 0]
        while True:  # Do - While loop
            for i in range(len(x)):
                next[i] = cur[i] - alf * grad(cur)[i]
            count += 1

            if func(next) - func(cur) < - alf * norma(grad(cur))**2 * 0.5:  # loop escape
                break
            else:
                alf *= lbd
        cur = next
        if norma(grad(cur)) < eps:  # loop escape
            break
    return [cur, count]


# Покоординатный спуск
def coordinate_wise_method(func, grad, x, eps):
    next = list(x)
    count = 0
    while True:
        cur = list(next)
        for i in range(len(x)):
            # Gold section method
            right = next[i]
            if -right < right:
                right *= 2.0
            else:
                right = -right*2.0
            left = -right
            if right - left < 2*eps:
                next[i] = (right + left) / 2.0
                break
            c_arg = list(next)
            c = left + (right - left) * (3 - 5**.5) / 2.0
            c_arg[i] = c
            count += 1
            d_arg = list(next)
            d = right - (right - left) * (3 - 5**.5) / 2.0
            d_arg[i] = d
            count += 1
            fc, fd = func(c_arg), func(d_arg)
            while right - left > 2*eps:
                if fc < fd:
                    right = d
                    if right - left < 2*eps:
                        break
                    d, fd = c, fc
                    c = left + (right - left) * (3 - 5**.5) / 2.0
                    c_arg[i] = c
                    count += 1
                    fc = func(c_arg)
                else:
                    left = c
                    if right - left < 2*eps:
                        break
                    c, fc = d, fd
                    d = right - (right - left) * (3 - 5**.5) / 2.0
                    d_arg[i] = d
                    count += 1
                    fd = func(d_arg)
            next[i] = (left + right) / 2.0
        if abs(func(cur) - func(next)) < eps:
            break
    return [next, count]


class Expr(object):
    def __init__(self):
        pass

    def is_true(self):
        return True


def mngs(func, grad, x, eps, expr):
    cur = list(x)
    count = 1
    f_cur = func(x)
    # Do - While loop
    while expr.is_true():
        f_prev = f_cur
        p, c = gold_section_in_space(func, cur, eps**2, grad(cur))
        count += c
        cur = p
        f_cur = func(p)
        count += 1
        if norma(grad(cur)) < eps:  # or f_prev - f_cur < eps:  # loop escape
            break

    return [cur, count]


# Наискорейший градиентный спуск
def fast_gradient(func, grad, x, eps):
    return mngs(func, grad, x, eps, Expr())


# Градиентный спуск по расходящимуся ряду ?
def convergent_series(func, grad, x, eps):
    cur = list(x)
    n = 0
    count = 0
    # Do - While loop
    while True:
        n += 1
        direction = list(grad(cur))
        for i in range(len(x)):
            cur[i] -= 1.0 / n * direction[i]
        count += 1
        if norma(grad(cur)) < eps:  # loop escape
            break
    return [cur, count]


class ExprCount(Expr):
    def __init__(self, cur, top):
        Expr.__init__(self)
        self.cur = cur
        self.top = top

    def is_true(self):
        ans = self.cur < self.top
        self.cur += 1
        return ans


def fastest_grad_method_p(func, grad, x, eps):
    count = 0
    cur = list(x)
    while True:
        mn = mngs(func, grad, cur, eps, ExprCount(0, len(x)))
        count += mn[1]
        step_dir = list()
        for i, val in enumerate(cur):
            step_dir.append(mn[0][i] - cur[i])
        cur, c = gold_section_in_space(func, mn[0], eps**2, step_dir)
        count += c
        if norma(grad(cur)) < eps:  # loop escape
            break

    return [cur, count]


def ravine_method(func, grad, x, eps):
    count = 0
    cur = list(x)
    cur_n = list(cur)
    while True:
        mn1 = mngs(func, grad, cur, eps*eps, ExprCount(0, len(x) - 1))
        if norma(grad(cur)) < eps:
            break
        for i, it in enumerate(cur_n):
            cur_n[i] += eps
        mn2 = mngs(func, grad, cur_n, eps*eps, ExprCount(0, len(x) - 1))
        count += mn1[1] + mn2[1]
        direction = list(mn1[0])
        for i, it in enumerate(direction):
            direction[i] -= mn2[0][i]
        cur, count = gold_section_in_space(func, mn1[0], eps*eps, direction)
        if norma(grad(cur)) < eps:
            break

    return [cur, count]


# func using as second grad
def newton_method(reverse, grad, x, eps):
    count = 0
    cur = list(x)
    while True:
        r = reverse(cur)
        f1 = numpy.array(r)
        f2 = numpy.array(grad(cur))
        count += 2
        step = f1.__matmul__(f2)
        for i, it in enumerate(cur):
            cur[i] -= step[i]
        if norma(grad(cur)) < eps:
            break
    return [cur, count]


def quasi_newton(func, grad, x, eps):
    count = 0
    H = Matrix(ones(len(x)))
    cur = list(x)
    iter = 0
    while True:
        last = list(cur)
        iter += 1
        direction = H * Matrix(make_matrix(grad(cur)))
        cur, c = gold_section_in_space(func, cur, eps*eps, make_vector(direction.data))
        count += c
        if norma(grad(cur)) < eps:  # Exit
            return [cur, count]

        if iter%(len(x) + 1) == 0:
            H = Matrix(ones(len(x)))
        else:
            tx = Matrix(make_matrix(cur)) - Matrix(make_matrix(last))
            tdx = Matrix(make_matrix(grad(cur))) - Matrix(make_matrix(grad(last)))
            F = tx - H * tdx
            A = F * F.transpose()
            B = F.transpose() * tdx
            H += A * (1.0 / B.data[0][0])


def conjugate_gradient(func, grad, x, eps):
    count = 0
    cur = list(x)
    d = grad(cur)
    b = 0
    iter = 0
    while True:
        iter += 1
        last = list(cur)
        cur, c = gold_section_in_space(func, cur, eps*eps, d)
        count += c
        if norma(grad(cur)) < eps:  # Exit
            return [cur, count]

        if iter%(len(x)*2) == 0:
            b = 0
        else:
            b = norma(grad(cur))**2 / norma(grad(last))**2
            count += 2

        #  d = -grad(cur) + b * d
        t = list(d)
        for i, it in enumerate(t):
            it *= b
        g = grad(cur)
        count += 1
        for i, it in enumerate(d):
            d[i] = t[i] + g[i]



allmethods = [
    gradient_fix_step,
    gradient_change_step,
    coordinate_wise_method,
    fast_gradient,
    #convergent_series
    fastest_grad_method_p,
    ravine_method,
    newton_method,
    quasi_newton,
    conjugate_gradient,
]


def run_method(output, method, func, grad, beg, eps):
    print("## Method: ", method.__name__, file=output)
    start = time.time()
    try:
        ans = method(func, grad, beg, eps)
        print("__Answer__:  ", file=output)
        print("`Xmin` =", ans[0], end="  \n", file=output)
        print("`F(Xmin)` = ", func(ans[0]), end="  \n", file=output)
        print("__Performance__:  ", file=output)
        print("`func calls` =", ans[1], end="  \n", file=output)
        print("`time` = ", round((time.time() - start)*1000, 3), "(ms)", file=output)
    except Exception as ex:
        print("__Error__:`", ex, "`", file=output)
    print("\n\n", file=output)


def run_all_methods(output, func, grad, dgrad, beg: list, eps: float):
    print("# ", func.__name__, "  \nstart from ", beg, "eps=", eps, end="  \n\n---  \n\n", file=output)
    for each in allmethods:
        if each is newton_method:
            run_method(output, each, dgrad, grad, beg, eps)
        else:
            run_method(output, each, func, grad, beg, eps)


if __name__ == '__main__':
    file = open('multidmethods.md', 'w')
    run_all_methods(file, f17102, f17102_g, f17102_dg, [2, 2], 0.001)
    run_all_methods(file, fsqr, fsqrt_g, fsqrt_dg, [2, 2], 0.001)