"""
Provide unit tests for LeastSquare class
"""

import leastSquareReg as lsr
import numpy as np
from leastSquareReg import DimensionMismatch

## test contructor
def testConstructor():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    mod1 = lsr.LeastSquare(x1,y1)
    expectedX = np.array([[1,1],[1,2],[1,3]])
    if np.array_equal(mod1.getX(), expectedX):
        print "constructed X CORRECT!"
    expectedY = np.array([[1],[2],[3]])
    if np.array_equal(mod1.getY(), expectedY):
        print "constructed y CORRECT!"
    x2 = np.array([1,2,3,4])
    try:
        mod2 = lsr.LeastSquare(x2,y1)
        print "FAILED to check dimensions!"
    except DimensionMismatch:
        print "dimension check CORRECT!"

def testNormFunc():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    mod1 = lsr.LeastSquare(x1, y1)
    estimator = mod1.normFunc()
    epson = 1e-6
    threshold = epson*np.ones_like(estimator)
    expEstimator = np.array([[0],[1]])
    if np.less(abs(estimator-expEstimator),threshold).all():
        print "normal function CORRECT!"
    else:
        print "FAILED normal function!"
    x2 = np.array([3,4,5])
    mod2 = lsr.LeastSquare(x2, y1)
    try:
        mod2.getNormFuncEst()
        print "FAILED to check field variable initialization!"
    except ValueError:
        print "check field variable initialization CORRECT!"
##--------Working in process---------
def testPredY():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    x2 = np.array([2,3,4])
    mod1 = lsr.LeastSquare(x1,y1)
    try:
        mod1.predY(estimator='NormalFunction')
        print "FAILED to catch non-initialization error!"
    except ValueError:
        print "check field variable initialization CORRECT!"
    mod1.normFunc()
    normfuncEst = mod1.getNormFuncEst()
    epson = 1e-6
    threshold = epson * y1
    estimatedY = mod1.predY(estimator='NormalFunction')
    y1vert = np.reshape(y1,(-1,1))
    if np.less(abs(y1vert-estimatedY),threshold).all():
        print "prediction of y CORRECT!"
    else:
        print "FAILED to correctly predict y!"
    try:
        estimatedY = mod1.predY(estimator='abc')
        print "FAILED to check validity of estimator!"
    except ValueError:
        print "check validity of estimator CORRECT!"
    estimatedY2 = mod1.predY(x2,'NormalFunction')
    y2vert = np.array([[2],[3],[4]])
    if np.less(abs(y2vert-estimatedY2), threshold).all():
        print "prediction of y from test set CORRECT!"
    else:
        print "FAILED to correctly predict y from test set!"
    

def testCompRSS():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    mod1 = lsr.LeastSquare(x1,y1)
    try:
        mod1.compRSS(x1, estimator="NormalFunction")
        print "FAILED to check input arguments!"
    except ValueError:
        print "check input arguments CORRECT!"
    try:
        mod1.compRSS(estimator='NormalFunction')
        print "FAILED to catch non-initialization error"
    except ValueError:
        print "check field variable initialization CORRECT!"
    mod1.normFunc()
    rss = mod1.compRSS(estimator='NormalFunction')
    epson = 1e-6
    if np.less(abs(rss), epson).all():
        print "compute of RSS CORRECT!"
    else:
        print "FAILED to compute RSS correctly!"
    return


if __name__ == '__main__':
    testConstructor()
    testNormFunc()
    testPredY()
    testCompRSS()


