import matplotlib.pyplot as plt
import numpy as np
import scipy.special

from sympy import Matrix, init_printing
from IPython.display import display

###################################################################

def showmat(A, numdig=None):
    if numdig is not None:
        if numdig == 0:
            A_rounded = np.around(A).astype(int)
        else:
            A_rounded = np.around(A, decimals=numdig)
    else:
        A_rounded = A
    sym_matrix = Matrix(A_rounded)
    display(sym_matrix)

###################################################################

# calculate the correlation matrix from the covariance matrix
#    rho = np.zeros((nparm,nparm))
#    for i in range(nparm):
#        for j in range(nparm):
#            rho[i,j] = C[i,j]/np.sqrt(C[i,i]*C[j,j])
#
def corrcov(C):
    nx,ny = C.shape
    if nx != ny:
        return
        
    # c = np.sqrt(np.diag(C)).reshape(nparm,1)
    # Crho = C/(c@c.T)
    sigma = np.sqrt(np.diag(C))
    outer_v = np.outer(sigma,sigma)
    Crho = C / outer_v
    
    Crho[C == 0] = 0
    return Crho
    
###################################################################

def plot_ellipse(DELTA2,C,m):
    # DELTA2 controls the size of the ellipse (see chi2inv in lib_peip.py)
    # C      2 x 2 input covariance matrix
    # m      2 x 1 (xs,ys) defining the center of the ellipse

    # construct a vector of n equally-spaced angles from (0,2*pi)
    n = 1000
    theta = np.linspace(0,2*np.pi,n).T
    # points defining unit circle
    xhat = np.array([np.cos(theta),np.sin(theta)]).T
    Cinv = np.linalg.inv(C)

    r = np.zeros((n,2))
    for i in range(n):
        #store each (x,y) pair on the confidence ellipse in the corresponding row of r
        #r(i,:) = sqrt(DELTA2/(xhat(i,:)*Cinv*xhat(i,:)'))*xhat(i,:)
        #r[i,:] = np.dot(np.sqrt(DELTA2/(xhat[i,:]@Cinv@xhat[i,:].T)),xhat[i,:])
        rlen   = np.sqrt(DELTA2 / (xhat[i,:] @ Cinv @ xhat[i,:].T))
        r[i,:] = rlen * xhat[i,:]
    
    # shift ellipse based on centerpoint m = (xs,ys)
    x = m[0] + r[:,0]
    y = m[1] + r[:,1]
    #plt.plot(x,y)
    
    return x,y
    
###################################################################

def phi(x):
    # Parameter Estimation and Inverse Problems, 3rd edition, 2018
    # by R. Aster, B. Borchers, C. Thurber
    # z=phi(x)
    #
    # Calculates the normal distribution and returns the value of the
    # integral
    #
    #       z=int((1/sqrt(2*pi))*exp(-t^2/2),t=-infinity..x)
    #
    # Input Parameters:
    #   x - endpoint of integration (scalar)
    #
    # Output Parameters:
    #   z - value of integral
    #Python version coded by Yuan Tian @UAF 2021
    if (x >= 0):
        z = 0.5 + 0.5 * scipy.special.erf(x/np.sqrt(2))
    else:
        z = 1 - phi(-x)
    return z


def chi2cdf(x,m):
    # Parameter Estimation and Inverse Problems, 3rd edition, 2018
    # by R. Aster, B. Borchers, C. Thurber
    # p=chi2cdf(x,m)
    #
    # Computes the Chi^2 CDF, using a transformation to N(0,1) on page
    # 333 of Thistead, Elements of Statistical Computing.
    #
    # Input Parameters:
    #   x - end value of chi^2 pdf to integrate to. (scalar)
    #   m - degrees of freedom (scalar)
    #
    # Output Parameters:
    #   p - probability that Chi^2 random variable is less than or
    #       equal to x (scalar).
    #
    # Note that x and m must be scalars.

    #Python version coded by Yuan Tian @UAF 2021
    if x==(m-1):
        p=0.5
    else:
        z = (x - m + 2 / 3 - 0.08 / m) * np.sqrt((m - 1)*np.log((m - 1)/x)+x-(m - 1)) / np.abs(x-m+1);
        p = phi(z)
    return p


def chi2inv(p,nu):
    # Parameter Estimation and Inverse Problems, 3rd edition, 2018
    # by R. Aster, B. Borchers, C. Thurber
    # x=chi2inv(p,nu)
    #
    # Computes the inverse Chi^2 distribution corresponding to a given
    # probability that a Chi^2 random variable with the given degrees
    # of freedom is less than or equal to x.  Uses chi2cdf.m.
    #
    # Input Parameters:
    #   p - probability that Chi^2 random variable is less than or
    #       equal to x (scalar).
    #   nu - degrees of freedom (scalar)
    #
    # Output Parameters:
    #   x - corresponding value of x for given probability.
    #
    # Note that x and m must be scalars.
    # Special cases.
    #Python version coded by Yuan Tian @UAF 2021
    if (p >= 1.0):
        return np.inf
    elif (p == 0.0):
        return 0
    elif (p < 0):
        return -np.inf

    # find a window with a cdf containing p
    l = 0.0
    r = 1.0
    
    while (chi2cdf(r, nu) < p):
        l = r
        r = r * 2
    # do a binary search until we have a sufficiently small interval around x
    while (((r - l) / r) > 1.0e-5):
        m = (l + r) / 2
        if (chi2cdf(m, nu) > p):
            r = m
        else:
            l = m
    return (l+r)/2

###################################################################