import time

# F(X)= sqrt(1 + x[0]^2 + x[1]^2)
def f17102(x):
    return (1 + x[0]**2 + x[1]**2)**.5


def f17102_g(x):
    return [x[0] / (1 + x[0]**2 + x[1]**2)**.5, x[1] / (1 + x[0]**2 + x[1]**2)**.5]


def fsqr(x):
    return (x[0]-5)**2 + (x[1]-3)**4


def fsqrt_g(x):
    return [2*(x[0] - 5), 4*(x[1]-3)**3]


# ||X|| = sqrt(sum(x[i]**2))
def norma(x):
    ans = 0
    for i in range(len(x)):
        ans += x[i]**2
    return ans**0.5


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
        for i in range(2):
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


# Наискорейший градиентный спуск
def fast_gradient(func, grad, x, eps):
    cur = list(x)
    count = 0
    f_cur = func(x)
    # Do - While loop
    while True:
        f_prev = f_cur
        prev = list(cur)
        direction = list(grad(cur))
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
        count += 1

        # Copy right probe
        d_arg = list(right_p)
        d = right - (right - left) * (3 - 5**.5) / 2.0
        for i in range(len(d_arg)):
            d_arg[i] += d * direction[i]
        count += 1

        fc, fd = func(c_arg), func(d_arg)  # func values in probe points
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
        f_cur = func(cur)
        if norma(grad(cur)) < eps or f_prev - f_cur < eps:  # loop escape
            break

    return [cur, count]


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


def fastest_grad_method(func, grad, x, eps):
    pass


allmethods = [
    gradient_fix_step,
    gradient_change_step,
    coordinate_wise_method,
    fast_gradient,
    #convergent_series
]


def run_method(output, method, func, grad, beg, eps):
    print("#### Method: ", method.__name__, file=output)
    start = time.time()
    ans = method(func, grad, beg, eps)
    print("__Method values__:  ", file=file)
    print("Xmin =", ans[0], end="  \n", file=file)
    print("F(Xmin) = ", func(ans[0]), end="  \n", file=file)
    print("iterations =", ans[1], end="  \n", file=file)
    print("time = ", round((time.time() - start)*1000, 3), "(ms)", file=file)
    print("\n\n", file=file)


def run_all_methods(output, func, grad, beg, eps):
    print("## ", func.__name__, "  \nstart from ", beg, "eps=", eps, end="  \n\n---  \n\n", file=output)
    for each in allmethods:
        run_method(output, each, func, grad, beg, eps)


if __name__ == '__main__':
    file = open('multidmethods.md', 'w')
    run_all_methods(file, f17102, f17102_g, [2, 2], 0.001)
    run_all_methods(file, fsqr, fsqrt_g, [2, 2], 0.01)