import matplotlib.pyplot as plt
import numpy as np
import random


X = np.linspace(-6,6,num=100)

# add gaussian noise
mu = 0
sigma = 4

convex_functions = [X**2, X**4, X**2+X**4, np.abs(X**3), 0.5*X**4, 0.5*X**2]
title_names = ["x^2", "x^4", "x^2 + x^4", "abs(x^3)", "0.5x^4", "0.5x^2"]
fig, ax = plt.subplots(3,2)
i = 0
j = 0

eft  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 5      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for blank space between subplots
hspace = 0.5   # the amount of height reserved for white space between subplots
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
for y in convex_functions:
    noise = np.random.normal(mu, sigma, size=(y.shape))
    y_noisy = np.add(y,noise)
    ax[i,j].plot(X,y)
    ax[i,j].plot(X,y_noisy, "*")
    ax[i,j].set_title(title_names[2*i+j])
    if j < 1 :
        j = j+1
    elif i < 2:
        j = 0
        i = i+1
    else:
        print("error: exceed subplot range")

plt.show()

for y in convex_functions:
    plt.plot(X,y)
    plt.legend(title_names)

plt.show()


