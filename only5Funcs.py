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


def eqnFiveAdjPairs(EIn, *data):
    """

    :param EIn: trial eigenvalue, in the form of [re, im]
    :param data:  (n, g)
    :return:
    """
    n,g=data
    E = EIn[0] + 1j * EIn[1]
    adjPairsAll=return5AdjacentPairs(g,E)
    retValsCis=[]#in the order x2, x1
    retValsTrans=[]# in the order x1,x2
    retValsCisAnother=[]#in the order x2, x1, another branch
    retValsTransAnother=[]#in the order x1, x2, another branch

    #fill cis
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadrature(g,E,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCis.append(rstTmp)
    #fill trans
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadrature(g,E,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTrans.append(rstTmp)
    #fill cis another
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadratureAnotherBranch(g,E,x1Tmp,x2Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsCisAnother.append(rstTmp)
    #fill trans another
    for pairTmp in adjPairsAll:
        x2Tmp,x1Tmp=pairTmp
        intValTmp=integralQuadratureAnotherBranch(g,E,x2Tmp,x1Tmp)
        rstTmp=intValTmp-(n+1/2)*np.pi
        retValsTransAnother.append(rstTmp)

    retCombined=retValsCis+retValsTrans+retValsCisAnother+retValsTransAnother
    retSorted=sorted(retCombined,key=np.abs)
    root0=retSorted[0]
    return np.real(root0),np.imag(root0)


def computeOneSolutionWith5AdjPairs(inData):
    """

    :param inData: [n,g, Eest]
    :return: [n,g,re(E), im(E)]
    """
    n, g, Eest = inData
    eVecTmp=sopt.fsolve(eqnFiveAdjPairs,[np.real(Eest), np.imag(Eest)], args=(n, g), maxfev=100, xtol=1e-6)
    return [n,g,eVecTmp[0],eVecTmp[1]]