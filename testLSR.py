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
        print "constructed X correct!"
    expectedY = np.array([[1],[2],[3]])
    if np.array_equal(mod1.getY(), expectedY):
        print "constructed y correct!"
    x2 = np.array([1,2,3,4])
    try:
        mod2 = lsr.LeastSquare(x2,y1)
        print "Failed to check dimensions!"
    except DimensionMismatch:
        print "dimension check correct!"

def testNormFunc():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    mod1 = lsr.LeastSquare(x1, y1)
    estimator = mod1.normFunc()
    epson = 1e-6
    threshold = epson*np.ones_like(estimator)
    expEstimator = np.array([[0],[1]])
    if np.less(abs(estimator-expEstimator),threshold).all():
        print "normal function correct!"
    else:
        print "Failed normal function!"

if __name__ == '__main__':
    testConstructor()
    testNormFunc()


