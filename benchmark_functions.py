from math import sin
from math import sqrt
from math import exp
from math import pi
from math import fabs

functions_search_space = {'cross_in_tray': (-10.0,10.0)}

def cross_in_tray(x1, x2):
    return -0.0001 * ( fabs( sin(x1)*sin(x2) * exp( fabs(100-sqrt(x1*x1+x2*x2)/pi) ) ) + 1 )**0.1

