import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    y= np.array(y)

    N , D = X.shape

    # initialize the paramenters
    w =np.zeros(D)
    b= 0.0

    for _ in range(steps) :
    # linear comb
      z = np.dot(X,w)+b

      p =_sigmoid(z)

      dw =(1/N)*np.dot(X.T,(p-y))
      db = (1/N) * np.sum(p-y)

      w -= lr * dw
      b -= lr*db

    return w ,b
    pass