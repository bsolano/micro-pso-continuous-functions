from math import sin
from math import cos
from math import sqrt
from math import exp
from math import pi
from math import fabs

functions_search_space = {'cross_in_tray': (-10.0,10.0), 'drop_in_wave': (-5.12,5.12)}

def cross_in_tray(x1, x2):
    return -0.0001 * ( fabs( sin(x1)*sin(x2) * exp( fabs(100-sqrt(x1*x1+x2*x2)/pi) ) ) + 1 )**0.1

def drop_in_wave(x1, x2):
    return -(1+cos(12*sqrt(x1*x1+x2*x2)))/(0.5*(x1*x1+x2*x2)+2)