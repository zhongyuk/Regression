class leastSquareReg:
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
    import numpy as np
    ## Constructor
    def __init__(self, X, y): 
        if X.shape[0]!=y.shape[0]:
            raise DimensionMismatch()
        try:
            y.shape[1]
            self._y = y
        except IndexError as e:
            self._y = np.reshape(y, (-1,1))
        self._num = y.shape[0]
        self._X = np.concatenate((self._colOfOnes(),X), axis=1)
        
    ## API methods
    #### A getter method for constructed X
    def getX(self):
        return self._X

    #### Normal function solution
    def normFunc():

    #### Compute Residual Sum of Squares
    def compRSS():

    #### Compute cost functon
    def compCostFunc():

    #### Gradient Descent
    def gradientDescent():
    
    #### Predict outcome
    def predY():
    
    #### Compute P value
    def pValue():
    


    ## Helper methods
    def _colOfOnes(self):
        return np.ones((self._num,1))
        

class DimensionMismatch(Exception):
    """
    Raise DimenionMismatch Exception when explanatory variable and response variable
    have different number of instances.
    """
    def __init__(self):
        self.value = 'Explanatory variable and response variable have different number of observations!'
    def __str__(self):
        return repr(self.value)
    

