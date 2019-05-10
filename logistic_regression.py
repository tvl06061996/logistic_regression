# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

class LogisticRegression(object):
    
    def train_test_data_split(self, X, y, test_size):
        X_number_of_test = math.ceil(X.shape[0] * test_size)
        y_number_of_test = math.ceil(y.shape[0] * test_size)
        X_train = X[X_number_of_test:]
        X_test = X[:X_number_of_test:]
        y_train = y[y_number_of_test:]
        y_test = y[:y_number_of_test]
        return X_train, X_test, y_train, y_test
    
    def sigmoid(self, x):
      return (1 / (1 + np.exp(-x)))
  
    def calculate_cost(self, h, y):
        return -sum(np.multiply(y, np.log(h)) + np.multiply((1 - y),np.log(1-h)))
    
    def fit(self, X, y, alpha=0.01, interations=1000):
        self.W = np.array(np.ones(X.shape[1] + 1), dtype=np.int).T
        ones = np.ones((X.shape[0], 1))
        X = np.concatenate((ones, X), axis=1)
        cost_history = np.zeros(interations)
        for interation in range(interations):
            h = self.sigmoid(np.dot(X, self.W))
            loss = h - y
            gradient = np.dot(X.T, loss)
            self.W = self.W - alpha * gradient
            cost_history[interation] = self.calculate_cost(h, y)
        self.draw(cost_history)
    
    def draw(self, cost_history):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_ylabel('J(Theta)')
        ax.set_xlabel('Interactions')
        _ = ax.plot(range(1000), cost_history, 'b.')
        plt.show()
    
    def coef_(self):
        return self.W
    
    def predict(self, X):
        W = self.W
        y_predicts = []
        for x in X:
            y_predict = W[0]
            y_predicts.append(self.sigmoid(y_predict + sum([W[i + 1] * x[i] for i in range(len(x))])))
        return y_predicts


