# Regression
The purposes of developping this repo are...
* Practice building your own regression class.
* Demonstrate performing regression analysis using self-built regression class vs. pandas and scikit-learn
* A comparison and discussion among various regression methods through analysis examples

### About LeastSquare class
* It is developped under Python2 environment. It is assumed that data is imported into workspace by using pandas. Functionalities of LeastSquare is implemented upon numpy. Therefore both explantory variables and response variables are both converted into 2D matrix internally before performing computations.
* Two solutions for calculating least square estimators are implemented: 
  1) closed-form /normal function solution
  2) batch gradient descent solution - available termination conditions are a) specified iteration number; b) gradient threshold termination
