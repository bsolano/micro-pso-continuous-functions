from numba import njit, prange

from math import sin
from math import cos
from math import sqrt
from math import exp
from math import pi
from math import fabs
from math import floor

from numpy import inf

functions_search_space = {
    'beale': (-4.5, 4.5),
    'biggs_exp2': (0.0, 20.0),
    'biggs_exp3': (0.0, 20.0),
    'biggs_exp4': (0.0, 20.0),
    'biggs_exp5': (0.0, 20.0),
    'biggs_exp6': (0.0, 20.0),
    'cross_in_tray': (-10.0, 10.0),
    'drop_in_wave': (-5.12, 5.12),
    'dejong_f1': (-5.12, 5.12),
    'dejong_f2': (-2.048, 2.048),
    'dejong_f3': (-5.12, 5.12),
    'dejong_f4': (-1.28, 1.28),
    'dejong_f5': (-65.536,65.536),
    'rosenbrock30': (-5,10),
    'rastringin30': (-5.12,5.12),
    'griewank30': (-512,512),
    'karaboga_akay': [(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1),(0,100),(0,100),(0,100),(0,1)]
}

functions_solution = {
    'beale': [3, 0.5],
    'biggs_exp2': [1, 10],
    'biggs_exp3': [1, 10, 5],
    'biggs_exp4': [1, 10, 1, 5],
    'biggs_exp5': [1, 10, 1, 5, 4],
    'biggs_exp6': [1, 10, 1, 5, 4, 3],
    'cross_in_tray': [1.349406685353340, 1.349406608602084],
    'drop_in_wave': [0, 0],
    'dejong_f1': [0, 0, 0],
    'dejong_f2': [1, 1],
    'dejong_f3': [-5.12, -5.12, -5.12, -5.12, -5.12],
    'dejong_f4': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'dejong_f5': [-32,-32],
    'rosenbrock30': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    'rastringin30': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'griewank30': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'karaboga_akay': [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 1]
}


@njit(cache=True)
def beale(x1, x2, x3):
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x3**3)**2


@njit(cache=True)
def biggs_exp2(x1, x2):
    sum = 0
    for i in prange(1, 11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi)
        sum += ((exp(-x1*zi) - 5*exp(-x2*zi)) - yi)**2
    return sum


@njit(cache=True)
def biggs_exp3(x1, x2, x3):
    sum = 0
    for i in prange(1, 11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi)
        sum += ((exp(-x1*zi) - x3*exp(-x2*zi)) - yi)**2
    return sum


@njit(cache=True)
def biggs_exp4(x1, x2, x3, x4):
    sum = 0
    for i in prange(1, 11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi)
        sum += ((x3*exp(-x1*zi) - x4*exp(-x2*zi)) - yi)**2
    return sum


@njit(cache=True)
def biggs_exp5(x1, x2, x3, x4, x5):
    sum = 0
    for i in prange(1, 11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi) + 3*exp(-4*zi)
        sum += ((x3*exp(-x1*zi) - x4*exp(-x2*zi) + 3*exp(-x5*zi)) - yi)**2
    return sum


@njit(cache=True)
def biggs_exp6(x1, x2, x3, x4, x5, x6):
    sum = 0
    for i in prange(1, 11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi) + 3*exp(-4*zi)
        sum += ((x3*exp(-x1*zi) - x4*exp(-x2*zi) + x6*exp(-x5*zi)) - yi)**2
    return sum


@njit(cache=True)
def cross_in_tray(x1, x2):
    return -0.0001 * (fabs(sin(x1)*sin(x2) * exp(fabs(100-sqrt(x1*x1+x2*x2)/pi))) + 1)**0.1


@njit(cache=True)
def drop_in_wave(x1, x2):
    return -(1 + cos(12*sqrt(x1*x1+x2*x2))) / (0.5*(x1*x1+x2*x2) + 2)


@njit(cache=True)
def dejong_f1(x1, x2, x3):
    return x1**2 + x2**2 + x3**2


@njit(cache=True)
def dejong_f2(x1, x2):
    return 100 * (x1**2-x2**2)**2 + (1-x1)**2


@njit(cache=True)
def dejong_f3(x1, x2, x3, x4, x5):
    return floor(x1) + floor(x2) + floor(x3) + floor(x4) + floor(x5)


@njit(cache=True)
def dejong_f4(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30):
    return 1 * x1**4 + 2 * x2**4 + 3 * x3**4 + 4 * x4**4 + 5 * x5**4 + 6 * x6**4 + 7 * x7**4 + 8 * x8**4 + 9 * x9**4 + 10 * x10**4 + 11 * x11**4 + 12 * x12**4 + 13 * x13**4 + 4 * x14**14 + 15 * x15**4 + 16 * x16**4 + 17 * x17**4 + 18 * x18**4 + 19 * x19**4 + 20 * x20**4 + 21 * x21**4 + 22 * x22**4 + 23 * x23**4 + 24 * x24**4 + 25 * x25**4 + 26 * x26**4 + 27 * x27**4 + 28 * x28**4 + 29 * x29**4 + 30 * x30**4


@njit(cache=True)
def f(j, x1, x2):
    a = [[-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32,-32,-16,0,16,32],[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32]]
    return j + (x1-a[0][j-1])**6 + (x2-a[1][j-1])**6


@njit(cache=True)
def dejong_f5(x1, x2):
    sum = 0
    for j in prange(1,26):
        sum += 1/f(j,x1,x2)
    return 1 / (0.002 + sum)


def rosenbrock30(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30):
    sum = 0
    for i in prange(1,30):
        x_iplus1 = locals()['x'+str(i+1)]
        x_i = locals()['x'+str(i)]
        sum += 100*(x_iplus1-x_i**2)**2+(x_i-1)**2
    return sum


@njit(cache=True)
def rastringin30(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30):
    sum = 0
    for x_i in (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30):
        sum += x_i**2 - 10 * cos(2 * pi * x_i) + 10
    return sum


@njit(cache=True)
def griewank30(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30):
    sum = 0
    mul = 1
    for i, x_i in enumerate((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24, x25, x26, x27, x28, x29, x30), 1):
        sum += x_i**2
        mul *= cos(x_i / sqrt(i))
    return sum/4000 - mul + 1


@njit(cache=True)
def karaboga_akay(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    sum1 = 0
    for x_i in (x1, x2, x3, x4):
        sum1 += x_i

    sum2 = 0
    for x_i in (x1, x2, x3, x4):
        sum2 += x_i**2

    sum3 = 0
    for x_i in (x5, x6, x7, x8, x9, x10, x11, x12, x13):
        sum3 += x_i

    sum = 5*sum1 - 5*sum2 - 5*sum3

    if 2*x1 + 2*x2 + x10 + x11 - 10 > 0:
        return inf
    elif 2*x1 + 2*x3 + x10 + x12 - 10 > 0:
        return inf
    elif 2*x2 + 2*x3 + x11 + x12 - 10 > 0:
        return inf
    elif -8*x1 + x10 > 0:
        return inf
    elif -8*x2 + x11 > 0:
        return inf
    elif -8*x3 + x12 > 0:
        return inf
    elif -2*x4 - x5 + x10 > 0:
        return inf
    elif -2*x6 - x7 + x11 > 0:
        return inf
    elif -2*x8 - x9 + x12 > 0:
        return inf
    else:
        return sum