#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 22:08:23 2019

@author: floydesk
"""


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
X_train, X_test, y_train, y_test = train_test_split( mnist.data, mnist.target, test_size=0.25, random_state=42)
# Standardization
from sklearn.preprocessing import StandardScaler

stdsc = StandardScaler()
stdsc.fit(X_train)
X_train_std = stdsc.transform(X_train)
X_test_std = stdsc.transform(X_test)

# Optimize hyperparameters
mlp = MLPClassifier(max_iter=20, random_state=42, verbose=True, hidden_layer_sizes=(10,))
mlp.fit(X_train_std, y_train)
print("Accuracy on the test set: {:.2f}".format(mlp.score(X_test_std, y_test)))
print("Activation function used at the output layer: %s" % mlp.out_activation_)
print("Number of outputs at the output layer: %f" % mlp.n_outputs_)
print("List predicted classes at the output layer: %s" % mlp.classes_)
print("Training set loss: %s" % mlp.loss_)1