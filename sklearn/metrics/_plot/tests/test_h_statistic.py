
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_boston
from sklearn.datasets import load_wine
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris

from sklearn.metrics._plot.h_statistic import compute_friedman_h_statistic

from sklearn.utils._testing import ignore_warnings


def test_h_statistic_size():
    data = [load_diabetes, load_boston, load_wine, load_breast_cancer, load_iris]

    for func in data:
        with ignore_warnings():
            csv_data = func()
            h_statistic = compute_friedman_h_statistic(csv_data.data, csv_data.target)

        array_size = len(h_statistic)
        for row in h_statistic:
            if len(row) != array_size:
                assert False

    assert True


def test_h_statistic_non_zeros():
    data = [load_diabetes, load_boston, load_wine, load_breast_cancer, load_iris]

    for func in data:
        with ignore_warnings():
            csv_data = func()
            h_statistic = compute_friedman_h_statistic(csv_data.data, csv_data.target)
        
        for i in range(len(h_statistic)):
            for j in range(i + 1, len(h_statistic)):
                if h_statistic[i][j] == 0:
                    assert False

    assert True


def test_h_statistic_zeros():
    data = [load_diabetes, load_boston, load_wine, load_breast_cancer, load_iris]

    for func in data:
        with ignore_warnings():
            csv_data = func()
            h_statistic = compute_friedman_h_statistic(csv_data.data, csv_data.target)
        
        for i in range(len(h_statistic)):
            for j in range(i + 1):
                if h_statistic[i][j] != 0:
                    assert False

    assert True
