from math import sin
from math import cos
from math import sqrt
from math import exp
from math import pi
from math import fabs
from math import log

functions_search_space = {
    'beale': (-4.5, 4.5),
    'biggs_exp2': (0.0,20.0),
    'biggs_exp3': (0.0,20.0),
    'biggs_exp4': (0.0,20.0),
    'biggs_exp5': (0.0,20.0),
    'biggs_exp6': (0.0,20.0),
    'cross_in_tray': (-10.0,10.0),
    'drop_in_wave': (-5.12,5.12),
    'weibull': (0.0,250.0)
}

def beale(x1, x2, x3):
    return (1.5 - x1 + x1*x2)**2 + (2.25 - x1 + x1*x2**2)**2 + (2.625 - x1 + x1*x3**3)**2

def biggs_exp2(x1, x2):
    sum = 0
    for i in range(1,11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi)
        sum += ( (exp(-x1*zi) - 5*exp(-x2*zi)) - yi )**2
    return sum

def biggs_exp3(x1, x2, x3):
    sum = 0
    for i in range(1,11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi)
        sum += ( (exp(-x1*zi) - x3*exp(-x2*zi)) - yi )**2
    return sum

def biggs_exp4(x1, x2, x3, x4):
    sum = 0
    for i in range(1,11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi)
        sum += ( (x3*exp(-x1*zi) - x4*exp(-x2*zi)) - yi )**2
    return sum

def biggs_exp5(x1, x2, x3, x4, x5):
    sum = 0
    for i in range(1,11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi) + 3*exp(-4*zi)
        sum += ( (x3*exp(-x1*zi) - x4*exp(-x2*zi) + 3*exp(-x5*zi)) - yi )**2
    return sum

def biggs_exp6(x1, x2, x3, x4, x5, x6):
    sum = 0
    for i in range(1,11):
        zi = 0.1*i
        yi = exp(-zi) - 5*exp(-10*zi) + 3*exp(-4*zi)
        sum += ( (x3*exp(-x1*zi) - x4*exp(-x2*zi) + x6*exp(-x5*zi)) - yi )**2
    return sum

def cross_in_tray(x1, x2):
    return -0.0001 * ( fabs( sin(x1)*sin(x2) * exp( fabs(100-sqrt(x1*x1+x2*x2)/pi) ) ) + 1 )**0.1

def drop_in_wave(x1, x2):
    return -( 1 + cos( 12*sqrt(x1*x1+x2*x2) )) / ( 0.5*(x1*x1+x2*x2) + 2 )

def weibull(x1, x2, x3):
    sum = 0
    for i in range(1,100):
        zi = 0.1*i
        yi = 25 + (50 * log(1/zi))**(2/3)
        sum += (exp(-((yi - x3)**x2/x1)) - zi)**2 
    return sum