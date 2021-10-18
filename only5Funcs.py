import numpy as np
import scipy.optimize as sopt
import matplotlib.pyplot as plt
from datetime import datetime
from multiprocessing import Pool
import scipy.special as sspecial
import mpmath
from mpmath import mp
mp.dps=100


#this script contains functions for V(x)=-igx^{5}
def return5AdjacentPairs(g,E):
    """

    :param g: const
    :param E: trial eigenvalue
    :return: 5 adjacent pairs of roots, the first root x2 has smaller angle than the second root x1,
    return [[x2, x1]]
    """
    coefs=[1j*g,0,0,0,0,E]
    rootsAll=np.roots(coefs)
    rootsSortedByAngle=sorted(rootsAll,key=np.angle)
    rst=[]
    length=len(rootsSortedByAngle)
    for j in range(0,length):
        rst.append([rootsSortedByAngle[j],rootsSortedByAngle[(j+1)%length]])

    return rst

def return5SeparatedPairs(g,E):
    """

    :param g: const
    :param E: trial eigenfunction
    :return: 5 pairs of roots separated by another root, the first root x2 has smaller angle than the
    second root x1, return [[x2, x1]]
    """
    coefs = [1j * g, 0, 0, 0, 0, E]
    rootsAll = np.roots(coefs)
    rootsSortedByAngle = sorted(rootsAll, key=np.angle)
    rst = []
    length = len(rootsSortedByAngle)
    for j in range(0, length):
        rst.append([rootsSortedByAngle[j], rootsSortedByAngle[(j + 2) % length]])

    return rst


def f(z,g,E):
    """

    :param z: point on x2x1
    :param g: const
    :param E: trial eigenvalue
    :return: (igx^{5}+E)^{1/2}
    """
    return (1j*g*z**5+E)**(1/2)

def fBranchAnother(z,g,E):
    """

    :param z: point on x2x1
    :param g:const
    :param E:trial eigenvalue
    :return: -(igx^{5}+E)^{1/2}
    """
    return -(1j*g*z**5+E)**(1/2)


def integralQuadrature(g,E,x1,x2):
    '''

    :param g: const
    :param E: trial eigenvalue
    :param x1: ending point
    :param x2: starting point
    :return:
    '''
    a1 = np.real(x1)
    b1 = np.imag(x1)

    a2 = np.real(x2)
    b2 = np.imag(x2)


    slope = (b1 - b2) / (a1 - a2)
    gFunc=lambda y:f(y+1j*(slope*(y-a2)+b2),g,E)
    return (1+1j*slope)*mpmath.quad(gFunc,[a2,a1])

def integralQuadratureAnotherBranch(g,E,x1,x2):
    '''

    :param g: const
    :param E: trial eigenvalue
    :param x1: ending point
    :param x2: starting point
    :return:
    '''
    a1 = np.real(x1)
    b1 = np.imag(x1)

    a2 = np.real(x2)
    b2 = np.imag(x2)

    slope = (b1 - b2) / (a1 - a2)
    gFunc = lambda y: fBranchAnother(y + 1j * (slope * (y - a2) + b2), g, E)
    return (1 + 1j * slope) * mpmath.quad(gFunc, [a2, a1])


def upperEigValue(g,n):
    """

    :param g: const
    :param n: energy level
    :return: eigenvalue from upper pair
    """
    return (5*(n+1/2)*np.pi*g**(1/5)/(
            2*np.cos(1/10*np.pi)
            *sspecial.beta(3/2,1/5)
                                      )
            )**(10/7)


def lowerEigValue(g,n):
    """

    :param g: const
    :param n: energy level
    :return: eigenvalue from lower pair
    """
    return (
        5*(n+1/2)*np.pi*g**(1/5)/(
        2*np.cos(3/10*np.pi)*sspecial.beta(1/5,3/2)
    )
    )**(10/7)