from numpy import *
from scipy import stats
from scipy import ndimage
import numpy as np
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False


def map_in_array(values):
    '''
    function to build arrays with correct shape for numpy.histogramdd()
    from 2 (or 3) time series of scalars
    '''
    if len(values) == 2:
        X = values[0]
        Y = values[1]
        data1 = list(map(lambda x, y: [x, y], X, Y))
        data = np.array(data1)
        return data
    if len(values) == 3:
        X = values[0]
        Y = values[1]
        Z = values[2]
        data1 = list(map(lambda x, y, z: [x, y, z], X, Y, Z))
        data = np.array(data1)
        return data


def transfer_entropy(X, Y, delay=1, gaussian_sigma=None):
    """
    TE implementation: asymmetric statistic measuring the reduction in uncertainty
    for a future value of X given the history of X and Y. Or the amount
    of information from Y to X. Calculated through the Kullback-Leibler divergence
    with conditional probabilities

    Args:
        X = time series of scalars (array or list)
        Y = time series of scalars (array or list)
    Kwargs:
        delay(int):
            step in tuple (x_n, y_n, x_(n - delay))
        gaussian_sigma(int):
            sigma to be used
            default set at None: no gaussian filtering applied
    Returns:
        float: Transfer Entropy between X and Y given the history of X
    """

    if len(X) != len(Y):
        raise ValueError('time series entries need to have same length')

    n = float(len(X[delay:]))

    # number of bins for X and Y using Freeman-Diaconis rule
    # histograms built with numpy.histogramdd

    binX = int((max(X) - min(X))
               / (2 * stats.iqr(X) / (len(X) ** (1.0 / 3))))
    binY = int((max(Y) - min(Y))
               / (2 * stats.iqr(Y) / (len(Y) ** (1.0 / 3))))

    p3, bin_p3 = histogramdd(
        sample=map_in_array([X[delay:], Y[:-delay], X[:-delay]]),
        bins=[binX, binY, binX]
    )
    p2, bin_p2 = histogramdd(
        sample=map_in_array([X[delay:], Y[:-delay]]),
        bins=[binX, binY]
    )
    p2delay, bin_p2delay = histogramdd(
        sample=map_in_array([X[delay:], X[:-delay]]),
        bins=[binX, binX]
    )
    p1, bin_p1 = histogramdd(
        sample=array(X[delay:]),
        bins=binX
    )

    # hists normalized to obtain densities
    p1 = p1 / n
    p2 = p2 / n
    p2delay = p2delay / n
    p3 = p3 / n

    # apply (or not) gaussian filters at given sigma to the distributions
    if gaussian_sigma is not None:
        s = gaussian_sigma
        p1 = ndimage.gaussian_filter(p1, sigma=s)
        p2 = ndimage.gaussian_filter(p2, sigma=s)
        p2delay = ndimage.gaussian_filter(p2delay, sigma=s)
        p3 = ndimage.gaussian_filter(p3, sigma=s)

    # ranges of values in time series
    Xrange = bin_p3[0][:-1]
    Yrange = bin_p3[1][:-1]
    X2range = bin_p3[2][:-1]

    # calculating elements in TE summation
    elements = []
    for i in range(len(Xrange)):
        px = p1[i]
        for j in range(len(Yrange)):
            pxy = p2[i][j]
            for k in range(len(X2range)):
                pxx2 = p2delay[i][k]
                pxyx2 = p3[i][j][k]

                arg1 = float(pxy * pxx2)
                arg2 = float(pxyx2 * px)
                # corrections avoding log(0)
                if arg1 == 0.0:
                    arg1 = float(1e-8)
                if arg2 == 0.0:
                    arg2 = float(1e-8)

                term = pxyx2 * log2(arg2) - pxyx2 * log2(arg1)
                elements.append(term)

    return sum(elements)
