import matplotlib.pyplot as plt
import numpy as np
import random
import pickle

def save_data(X,y,filename):
    f = open(filename, 'wb')
    pickle.dump(X,f)
    pickle.dump(y,f)
    f.close()
def graph_function_subplots(num_functions, X, y_functions, y_data, titles):
    fig = plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    for i in range(0,num_functions):
        ax = fig.add_subplot(3,2,i+1)

        ax.plot(X,y_functions[i])
        ax.plot(X,y_data[i], "*")
        ax.set_title(titles[i])

    plt.show()

def graph_all_functions(X,y, titles):
    fig = plt.figure()
    for y in convex_functions:
        plt.plot(X,y)
    
    plt.legend(titles)
    plt.show()

if __name__ == "__main__":
    # parameters
    num_functions = 6
    min_val = -5        # function boundaries
    max_val = 5
    mu = 0      # gaussian distribution params
    sigma = 4
    np.random.seed(1)   # set seed

    X = np.linspace(min_val,max_val,num=100)

    convex_functions = [X**2, X**4, X**2+X**4, np.abs(X**3), 0.5*X**4, 0.5*X**2]
    # add gaussian noise to each convex function
    noise = np.random.normal(mu, sigma, size=(convex_functions[0].shape))
    noisy_functions = [np.add(y, noise) for y in convex_functions]

    save_data(X,noisy_functions, "2d_data.pkl")
    titles = ["x^2", "x^4", "x^2 + x^4", "abs(x^3)", "0.5x^4", "0.5x^2"]

    graph_function_subplots(num_functions, X, convex_functions, noisy_functions, titles)
    graph_all_functions(X,convex_functions,titles)
