from numpy import *
class LeastSquare:
    """
    My own least square regression class.
    
    API Methods:
    getX(): return constructed X matrix
    normFunc():
    compRSS():
    compCostFunc():
    gradientDescent():
    predY():
    
    Attributes:
    _X:  the explanatory variable matrix after padding a column of ones
    _y:  the response variable vector
    _num:   num of observations
    _normFuncEst: least square estimator computed from normal function
    _gradDescEst: least square estimator computed from gradient descent

    """
    
    ## Constructor
    def __init__(self, X, y):
        self._num = y.shape[0]
        self._normFuncEst = None
        self._gradDescEst = None
        if X.shape[0]!=y.shape[0]:
            raise DimensionMismatch()
        try:
            y.shape[1]
            self._y = y
        except IndexError:
            self._y = reshape(y, (-1,1))
        ## for simple linear regression
        try:
            X.shape[1]
            self._X = concatenate((self._colOfOnes(),X), axis=1)
        except IndexError:
            x2D = reshape(X, (-1,1))
            self._X = concatenate((self._colOfOnes(),x2D), axis=1)
        
    ## API methods
    #### Getter methods
    def getX(self):
        return self._X
    
    def getY(self):
        return self._y
    
    def getNormFuncEst(self):
        if self._normFuncEst is None:
            message = 'Normal Function Calculated Least Square Estimator has not been initialized yet!'
            raise ValueError(message)
        else:
            return self._normFuncEst

    def getGradDescEst(self):
        if self._gradDescEst is None:
            message = 'Gradient Descent Calculated Least Square Estimator has not been initialized yet!'
            raise ValueError(message)
        else:
            return self._gradDescEst

    #### Normal function solution
    def normFunc(self):
        self._normFuncEst = dot(dot(linalg.inv(dot(self._X.T, self._X)),self._X.T), self._y)
        return self.getNormFuncEst()## The estimator

    #### Compute Residual Sum of Squares
    def compRSS(self, newX=None, newY=None, estimator='GradientDescent'):
        if (newX is None) and (newY is None):
            newY = self._y
        elif (newX is not None) and (newY is not None):
            if newX.shape[0]!=newY.shape[0]:
                raise DimensionMismatch()
            try:
                newY.shape[1]
            except IndexError:
                newY = reshape(newY, (-1,1))
        else:
            message = 'newX and newY need to be both specified at the same time, or NOT specified at the same time.'
            raise ValueError(message)
        if estimator=='GradientDescent':
            if self._gradDescEst is None:
                message = 'Gradient Descent Calculated Least Square Estimator has not been initialized yet!'
                raise ValueError(message)
            else:
                rss = sum(square(self.predY(newX, estimator)-newY))
        elif estimator=='NormalFunction':
            if self._normFuncEst is None:
                message = 'Normal Function Calculated Least Square Estimator has not been initialized yet!'
                raise ValueError(message)
            else:
                rss = sum(square(self.predY(newX, estimator)-newY))
        else:
            message = "Invalid Argument: estimator type can only be 'GradientDescent' or 'NormalFunction'"
            raise ValueError(message)
        return rss

    #### Compute cost functon
    def compCostFunc(self):
        return

    #### Gradient Descent
    def gradientDescent(self):
        return
    
    #### Predict outcome
    def predY(self, newX=None, estimator='GradientDescent'):
        if newX is None:
            newX = self._X
        else:
            try:
                newX.shape[1]
                newX = concatenate((self._colOfOnes(),newX), axis=1)
            except IndexError:
                newX = reshape(newX, (-1,1))
                newX = concatenate((self._colOfOnes(),newX), axis=1)
        if estimator=='GradientDescent':
            if self._gradDescEst is None:
                message = 'Gradient Descent Calculated Least Square Estimator has not been initialized yet!'
                raise ValueError(message)
            else:
                estimatedY = dot(newX, self._gradDescEst)
        elif estimator=='NormalFunction':
            if self._normFuncEst is None:
                message = 'Normal Function Calculated Least Square Estimator has not been initialized yet!'
                raise ValueError(message)
            else:
                estimatedY = dot(newX, self._normFuncEst)
        else:
            message = "Invalid Argument: estimator type can only be 'GradientDescent' or 'NormalFunction'"
            raise ValueError(message)
        return estimatedY
    
    #### Compute P value
    def pValue():
        return
    


    ## Helper methods
    def _colOfOnes(self):
        return ones((self._num,1))
        

class DimensionMismatch(Exception):
    """
    Raise DimenionMismatch Exception when explanatory variable and response variable
    have different number of instances.
    """
    def __init__(self):
        self.value = 'Explanatory variable and response variable have different number of observations!'
    def __str__(self):
        return repr(self.value)

    

