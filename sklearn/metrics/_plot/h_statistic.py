
import itertools as iter
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import partial_dependence


def compute_friedman_h_statistic(data, target):
    rf = RandomForestRegressor(n_estimators=10).fit(data, target)
    h_statistic = np.zeros((data.shape[1], data.shape[1]))

    for x, y in iter.combinations(range(data.shape[1]), 2):

        # calculate partial dependence of x in random forest
        x_partial_dependence = partial_dependence(rf, data, features=[x], kind='average')['average']

        # calculate partial dependence of y in random forest
        y_partial_dependence = partial_dependence(rf, data, features=[y], kind='average')['average']

        # calculate sum of univariate partial dependences
        sum_univariate_partial_dependences = x_partial_dependence.reshape(1, -1, 1) + y_partial_dependence.reshape(1, 1, -1)
        
        # calculate partial dependence of pair (x, y) in random forest
        x_y_partial_dependence = partial_dependence(rf, data, features=[x, y], kind='average')['average']

        # calcualte h statistic for pair (x, y)
        numerator = ((x_y_partial_dependence - sum_univariate_partial_dependences + target.mean() ) ** 2)
        denominator = ((x_y_partial_dependence - target.mean()) ** 2)
        h_statistic[x, y] = numerator.sum() / denominator.sum()

    # generate a colour bar for the computed h statistic
    plt.matshow(h_statistic.squeeze())
    plt.colorbar()
    plt.waitforbuttonpress(0)

    return h_statistic
