"""
Provide unit tests for LeastSquare class
"""

import leastSquareReg as lsr
import numpy as np
from leastSquareReg import DimensionMismatch
from leastSquareReg import GradientDescentError

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
    epson = 1e-3
    threshold = epson * np.ones((3,1))
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
    gradDesEst = mod1.gradientDescent(step=0.05, iteration=150)
    gradDesEstY = mod1.predY()
    if np.less(abs(y1vert-gradDesEstY), threshold).all():
        print "gradient descent prediction of y CORRECT!"
    else:
        print "FAILED to predict y correctly using gradient descent!"
    

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
    rssNF = mod1.compRSS(estimator='NormalFunction')
    epson = 1e-6
    if np.less(abs(rssNF), epson).all():
        print "compute RSS through normal function CORRECT!"
    else:
        print "FAILED to compute RSS correctly through normal function!"
    mod1.gradientDescent(step=0.05, iteration=150)
    rssGD = mod1.compRSS()
    if np.less(abs(rssGD), epson).all():
        print "compute RSS through gradient descent CORRECT!"
    else:
        print "FAILED to compute RSS correctly through gradient descent!"

def testGradientDescent():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    mod1 = lsr.LeastSquare(x1,y1)
    step1 = 0.1
    iteration = 100
    try:
        theta, costs = mod1.gradientDescent(step=step1, iteration=iteration)
        print "FAILED to check gradient decreasing trend!"
    except GradientDescentError:
        print "check gradient decreasing trend CORRECT!"
    step2 = 0.05
    thetaGD, costs = mod1.gradientDescent(step=step2, iteration = iteration)
    thetaNormFunc = mod1.normFunc()
    if np.less(abs(thetaGD-thetaNormFunc),np.array([[0.05],[0.05]])).all():
        print "gradient descent with specified iteration solution CORRECT!"
    else:
        print "FAILED gradient descent with specified iteration solution, mismatch with normal function solution!"
    thetaGD2, costs2 = mod1.gradientDescent(step=step2)
    if np.less(abs(thetaGD2-thetaNormFunc), np.array([[0.05],[0.05]])).all():
        print "gradient descent with easy termination solution CORRECT!"
    else:
        print "FAILED gradient descent with easy termination solution!"
    mod1.visulizeCosts()
    
def testCompRSquare():
    x1 = np.array([1,2,3])
    y1 = np.array([1,2,3])
    mod1 = lsr.LeastSquare(x1,y1)
    step2 = 0.05
    iteration = 100
    thetaGD, costs = mod1.gradientDescent(step=step2, iteration = iteration)
    RSquare = mod1.compRSquare()
    if (RSquare - 1.) < 1e-3:
        print "compute determination of coefficient CORRECT!"
    else:
        print "FAILED to compute the determination of coefficient within accuracy threshold!"

if __name__ == '__main__':
    testConstructor()
    testNormFunc()
    testPredY()
    testCompRSS()
    testGradientDescent()
    testCompRSquare()


