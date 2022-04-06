from math import sin
from math import cos
from math import sqrt
from math import exp
from math import pi
from math import fabs

functions_search_space = {'cross_in_tray': (-10.0,10.0), 'drop_in_wave': (-5.12,5.12), 'biggs_exp4': (0,20)}

def cross_in_tray(x1, x2):
    return -0.0001 * ( fabs( sin(x1)*sin(x2) * exp( fabs(100-sqrt(x1*x1+x2*x2)/pi) ) ) + 1 )**0.1

def drop_in_wave(x1, x2):
    return -( 1 + cos( 12*sqrt(x1*x1+x2*x2) )) / ( 0.5*(x1*x1+x2*x2) + 2 )

def biggs_exp4(x1, x2, x3, x4):
    sum = 0
    for i in range(1,11):
        ti = 0.1*i
        yi = exp(-ti) - 5*exp(-10*ti)
        sum += ( x3*exp(-ti*x1) - x4*exp(-ti*x2) - yi )**2
    return sum
    #return sum([(x3 * exp(-0.1*i*x1) - x4*exp(-0.1*i*x2) - (exp(-0.1*i) - 5*exp(-a10*0.1*i)))**2 for i in range(1,11)])