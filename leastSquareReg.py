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

    """
    
    ## Constructor
    def __init__(self, X, y):
        self._num = y.shape[0]
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
    #### A getter method for constructed X, y
    def getX(self):
        return self._X
    
    def getY(self):
        return self._y

    #### Normal function solution
    def normFunc(self):
        theta = dot(dot(linalg.inv(dot(self._X.T, self._X)),self._X.T), self._y)
        return theta ## The estimator

    #### Compute Residual Sum of Squares
    def compRSS(self):
        return

    #### Compute cost functon
    def compCostFunc():
        return

    #### Gradient Descent
    def gradientDescent():
        return
    
    #### Predict outcome
    def predY():
        return
    
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
    

