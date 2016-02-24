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
        if X.shape[0]!=y.shape[0]:
            raise DimensionMismatch()
        try:
            y.shape[1]
            self._y = y
        except IndexError:
            self._y = reshape(y, (-1,1))
        self._num = y.shape[0]
        self._X = concatenate((self._colOfOnes(),X), axis=1)
        
    ## API methods
    #### A getter method for constructed X
    def getX(self):
        return self._X

    #### Normal function solution
    def normFunc():
        return

    #### Compute Residual Sum of Squares
    def compRSS():
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
    

