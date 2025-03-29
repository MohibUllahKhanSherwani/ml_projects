import numpy as np
class Linear_Regression():
  # Initiating the hyperparameters
  def __init__(self, learning_rate, no_of_iters):
    self.learning_rate = learning_rate
    self.no_of_iters = no_of_iters
  def fit(self, x, y):
    # No. of training examples No. of features(columns)
    self.m, self.n = x.shape
    # Initializing weights
    self.w = np.zeros(self.n) # An array of zeros of size n(columns)
    # Initializing bias
    self.b = 0

    self.x = x
    self.y = y

    # Implementing gradient descent
    for i in range(self.no_of_iters):
      self.update_weights()

  def update_weights(self,):
    y_pred = self.predict(self.x)
    # Calculating gradients
    dw = -(2 * (self.x.T).dot(self.y - y_pred)) / self.m
    db = - 2 * np.sum(self.y - y_pred) / self.m
    # Updating weights
    self.w = self.w - self.learning_rate * dw
    self.b = self.b - self.learning_rate * db
  def predict(self, x):
    # y = wX + b
    return x.dot(self.w) + self.b # dot product of (w and x) + b
