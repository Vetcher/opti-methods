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
    # Do - While loop
    while True:
        for i in range(len(x)):
            cur[i] -= alf * grad(cur)[i]
        if norma(grad(cur)) < eps:  # loop escape
            break
    return cur


# Градиентный спуск с дроблением шага
def gradient_change_step(func, grad, x, eps):
    cur = list(x)
    lbd = 0.5
    # Do - While loop
    while True:
        alf = 10
        next = [0, 0]
        while True:  # Do - While loop
            for i in range(len(x)):
                next[i] = cur[i] - alf * grad(cur)[i]

            if func(next) - func(cur) < - alf * norma(grad(cur))**2 * 0.5:  # loop escape
                break
            else:
                alf *= lbd
        cur = next
        if norma(grad(cur)) < eps:  # loop escape
            break
    return cur


# Покоординатный спуск
def coordinate_wise_method(func, x, eps):
    next = list(x)
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
            d_arg = list(next)
            d = right - (right - left) * (3 - 5**.5) / 2.0
            d_arg[i] = d
            fc, fd = func(c_arg), func(d_arg)
            while right - left > 2*eps:
                if fc < fd:
                    right = d
                    if right - left < 2*eps:
                        break
                    d, fd = c, fc
                    c = left + (right - left) * (3 - 5**.5) / 2.0
                    c_arg[i] = c
                    fc = func(c_arg)
                else:
                    left = c
                    if right - left < 2*eps:
                        break
                    c, fc = d, fd
                    d = right - (right - left) * (3 - 5**.5) / 2.0
                    d_arg[i] = d
                    fd = func(d_arg)
            next[i] = (left + right) / 2.0
        if abs(func(cur) - func(next)) < eps:
            break
    return next


# Наискорейший градиентный спуск
def fast_gradient(func, grad, x, eps):
    cur = list(x)
    lbd = 0.5
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

        f_for_delta = func(prev)  # func value for new point

        while f_for_delta <= f_prev:
            delta = delta * 1.3 + delta**.5  # variate this function
            for i in range(len(prev)):  # do new step
                prev[i] = old[i] - delta*direction[i]
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
        while abs(right - left) > 2*eps:  # gold section loop
            if fc < fd:
                right = d
                if abs(right - left) < 2*eps:
                    break
                d, fd = c, fc
                c = left + (right - left) * (3 - 5**.5) / 2.0
                c_arg[i] = c
                fc = func(c_arg)
            else:
                left = c
                if abs(right - left) < 2*eps:
                    break
                c, fc = d, fd
                d = right - (right - left) * (3 - 5**.5) / 2.0
                d_arg[i] = d
                fd = func(d_arg)
        delta = (right + left) / 2.0
        for i in range(len(cur)):
            cur[i] = old[i] - delta * direction[i]
        f_cur = func(cur)
        if norma(grad(cur)) < eps or f_prev - f_cur < eps:  # loop escape
            break

    return cur


# Градиентный спуск по расходящимуся ряду ?
def convergent_series(func, grad, x, eps):
    cur = list(x)
    n = 0
    # Do - While loop
    while True:
        n += 1
        for i in range(len(x)):
            cur[i] -= 1.0 / n * grad(cur)[i]
        if norma(grad(cur)) < eps:  # loop escape
            break
    return cur


def run_all_methods(file, func, grad, beg, eps):
    print("## F(X)= sqrt(1 + x[0]^2 + x[1]^2)", end="  \n\n---  \n\n", file=file)
    ans = coordinate_wise_method(func, beg, eps)
    print("Coordinate:  ", file=file)
    print("Xmin =", ans, end="  \n", file=file)
    print("F(Xmin) = ", func(ans), file=file)
    print("\n\n", file=file)
    ans = gradient_fix_step(func, grad, beg, eps)
    print("Gradient with fixed step:  ", file=file)
    print("Xmin =", ans, end="  \n", file=file)
    print("F(Xmin) = ", func(ans), file=file)
    print("\n\n", file=file)
    ans = gradient_change_step(func, grad, beg, eps)
    print("Gradient with changing step:  ", file=file)
    print("Xmin =", ans, end="  \n", file=file)
    print("F(Xmin) = ", func(ans), file=file)
    print("\n\n", file=file)
    ans = fast_gradient(func, grad, beg, eps)
    print("Fastest gradient method:  ", file=file)
    print("Xmin =", ans, end="  \n", file=file)
    print("F(Xmin) = ", func(ans), file=file)
    '''
    print("\n\n", file=file)
    ans = convergent_series(func, grad, beg, eps)
    print("Convergent series:  ", file=file)
    print("Xmin =", ans, end="  \n", file=file)
    print("F(Xmin) = ", func(ans), file=file)
    '''


if __name__ == '__main__':
    file = open('test.md', 'w')
    run_all_methods(None, f17102, f17102_g, [2, 2], 0.001)
    #run_all_methods(None, fsqr, fsqrt_g, [2, 2], 0.01)