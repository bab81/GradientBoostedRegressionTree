import numpy as np
from numpy.matlib import repmat
import sys
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
from helper import *

print('You\'re running python %s' % sys.version.split(' ')[0])

xTrSpiral,yTrSpiral,xTeSpiral,yTeSpiral= spiraldata(150)
xTrIon,yTrIon,xTeIon,yTeIon= iondata()

# Create a regression tree with depth 4
tree = RegressionTree(depth=4)

# To fit/train the regression tree
tree.fit(xTrSpiral, yTrSpiral)

# To use the trained regression tree to predict a score for the example
score = tree.predict(xTrSpiral)

# To use the trained regression tree to make a +1/-1 prediction
pred = np.sign(tree.predict(xTrSpiral))

# Evaluate the depth 4 decision tree
print("Training error: %.4f" % np.mean(np.sign(tree.predict(xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(tree.predict(xTeSpiral)) != yTeSpiral))


def visclassifier(fun, xTr, yTr, newfig=True):
    """
    visualize decision boundary
    Define the symbols and colors we'll use in the plots later
    """

    yTr = np.array(yTr).flatten()

    symbols = ["ko", "kx"]
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    # get the unique values from labels array
    classvals = np.unique(yTr)

    if newfig:
        plt.figure()

    # return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(min(xTr[:, 0]), max(xTr[:, 0]), res)
    yrange = np.linspace(min(xTr[:, 1]), max(xTr[:, 1]), res)

    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # test all of these points on the grid
    testpreds = fun(xTe)

    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)
    # Z[0,0] = 1 # optional: scale the colors correctly

    # fill in the contours for these predictions
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    # creates x's and o's for training set
    for idx, c in enumerate(classvals):
        plt.scatter(xTr[yTr == c, 0],
                    xTr[yTr == c, 1],
                    marker=marker_symbols[idx],
                    color='k'
                    )

    plt.axis('tight')
    # shows figure and blocks
    plt.show()


visclassifier(lambda X: tree.predict(X), xTrSpiral, yTrSpiral)


def evalboostforest(trees, X, alphas=None):
    """Evaluates X using trees.

    Input:
        trees:  list of TreeNode decision trees of length m
        X:      n x d matrix of data points
        alphas: m-dimensional weight vector

    Output:
        pred: n-dimensional vector of predictions
    """
    m = len(trees)
    n, d = X.shape

    if alphas is None:
        alphas = np.ones(m) / len(trees)

    pred = np.zeros(n)

    for t in range(len(trees)):
        pred = pred + alphas[t] * trees[t].predict(X)

    return pred


def GBRT(xTr, yTr, m, maxdepth=4, alpha=0.1):
    """Creates GBRT.

    Input:
        xTr:      n x d matrix of data points
        yTr:      n-dimensional vector of labels
        m:        number of trees in the forest
        maxdepth: maximum depth of tree
        alpha:    learning rate for the GBRT


    Output:
        trees: list of decision trees of length m
        weights: weights of each tree
    """

    n, d = xTr.shape
    trees = []
    weights = []

    # Make a copy of the ground truth label
    # this will be the initial ground truth for our GBRT
    # This should be updated for each iteration
    t = np.copy(yTr)

    for i in range(m):
        tree = RegressionTree(maxdepth)
        tree.fit(xTr, t)
        trees.append(tree)
        weights.append(alpha)

        predH = evalboostforest(trees, xTr, weights)
        t = yTr - predH

    return trees, weights

trees, weights = GBRT(xTrSpiral,yTrSpiral, 50)

trees, weights=GBRT(xTrSpiral,yTrSpiral, 40, maxdepth=4, alpha=0.03) # compute tree on training data
visclassifier(lambda X:evalboostforest(trees, X, weights),xTrSpiral,yTrSpiral)
print("Training error: %.4f" % np.mean(np.sign(evalforest(trees,xTrSpiral)) != yTrSpiral))
print("Testing error:  %.4f" % np.mean(np.sign(evalforest(trees,xTeSpiral)) != yTeSpiral))

M=40 # max number of trees
err_trB=[]
err_teB=[]
alltrees, allweights =GBRT(xTrIon,yTrIon, M, maxdepth=4, alpha=0.05)
for i in range(M):
    trees=alltrees[:i+1]
    weights=allweights[:i+1]
    trErr = np.mean(np.sign(evalboostforest(trees,xTrIon, weights)) != yTrIon)
    teErr = np.mean(np.sign(evalboostforest(trees,xTeIon, weights)) != yTeIon)
    err_trB.append(trErr)
    err_teB.append(teErr)
    print("[%d]training err = %.4f\ttesting err = %.4f" % (i,trErr, teErr))

plt.figure()
line_tr, = plt.plot(range(M), err_trB, '-*', label="Training Error")
line_te, = plt.plot(range(M), err_teB, '-*', label="Testing error")
plt.title("Gradient Boosted Trees")
plt.legend(handles=[line_tr, line_te])
plt.xlabel("# of trees")
plt.ylabel("error")
plt.show()


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain, yTrain, w, b, M, Q, trees, weights

    if event.key == 'shift':
        Q += 10
    else:
        Q += 1
    Q = min(Q, M)

    classvals = np.unique(yTrain)

    # return 300 evenly spaced numbers over this interval
    res = 300
    xrange = np.linspace(0, 1, res)
    yrange = np.linspace(0, 1, res)

    # repeat this matrix 300 times for both axes
    pixelX = repmat(xrange, res, 1)
    pixelY = repmat(yrange, res, 1).T

    xTe = np.array([pixelX.flatten(), pixelY.flatten()]).T

    # get forest

    fun = lambda X: evalboostforest(trees[:Q], X, weights[:Q])
    # test all of these points on the grid
    testpreds = fun(xTe)
    trerr = np.mean(np.sign(fun(xTrain)) == np.sign(yTrain))

    # reshape it back together to make our grid
    Z = testpreds.reshape(res, res)

    plt.cla()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    # fill in the contours for these predictions
    marker_symbols = ['o', 'x']
    mycolors = [[0.5, 0.5, 1], [1, 0.5, 0.5]]
    plt.contourf(pixelX, pixelY, np.sign(Z), colors=mycolors)

    for idx, c in enumerate(classvals):
        plt.scatter(xTrain[yTrain == c, 0], xTrain[yTrain == c, 1], marker=marker_symbols[idx], color='k')
    plt.show()
    plt.title('# Trees: %i Training Accuracy: %2.2f' % (Q, trerr))


xTrain = xTrSpiral.copy() / 14 + 0.5
yTrain = yTrSpiral.copy()
yTrain = yTrain.astype(int)

# Hyper-parameters (feel free to play with them)
M = 50
alpha = 0.05;
depth = 5;
trees, weights = GBRT(xTrain, yTrain, M, alpha=alpha, maxdepth=depth)
Q = 0;


fig = plt.figure()
cid = fig.canvas.mpl_connect('button_press_event', onclick_forest)
print('Click to add a tree.');
plt.title('Click to start boosting on the spiral data.')
visclassifier(lambda X: np.sum(X, 1) * 0, xTrain, yTrain, newfig=False)
plt.xlim(0, 1)
plt.ylim(0, 1)


def onclick_forest(event):
    """
    Visualize forest, including new point
    """
    global xTrain, yTrain, Q, trees, weights

    if event.key == 'shift':
        Q += 10
    else:
        Q += 1
    Q = min(Q, M)

    plt.cla()
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    pTest = evalboostforest(trees[:Q], xTest, weights[:Q])
    pTrain = evalboostforest(trees[:Q], xTrain, weights[:Q])

    errTrain = np.sqrt(np.mean((pTrain - yTrain) ** 2))
    errTest = np.sqrt(np.mean((pTest - yTest) ** 2))

    plt.plot(xTrain[:, 0], yTrain, 'bx')
    plt.plot(xTest[:, 0], pTest, 'r-')
    plt.plot(xTest[:, 0], yTest, 'k-')

    plt.legend(['Training data', 'Prediction'])
    plt.title('(%i Trees)  Error Tr: %2.4f, Te:%2.4f' % (Q, errTrain, errTest))
    plt.show()
