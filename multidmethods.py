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
    ans = convergent_series(func, grad, beg, eps)
    print("Convergent series:  ", file=file)
    print("Xmin =", ans, end="  \n", file=file)
    print("F(Xmin) = ", func(ans), file=file)




if __name__ == '__main__':
    file = open('test.md', 'w')
    run_all_methods(file, f17102, f17102_g, [2, 2], 0.01)
    #run_all_methods(file, fsqr, fsqrt_g, [2, 2], 0.01)