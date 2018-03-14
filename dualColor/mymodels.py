import numpy as np
import scipy as sp
import scipy.optimize


def linear_regression(x, y, b0=None):
    x = np.array(x)
    y = np.array(y)
    if b0 is None:
        degrees = [0, 1]
    elif b0 == 0:
        degrees = [1]
    matrix = np.stack([x ** d for d in degrees], axis=-1)  # stack them like columns
    coeff = np.linalg.lstsq(matrix, y)[0]  # lstsq returns some additional info we ignore
    # print("Coefficients", coeff)
    yhat = np.dot(matrix, coeff)
    # print("Fitted curve/line", fit)
    return coeff, yhat


def fit_exponential(t, y, linear_method=None, c0=None, p0=None):

    # y(t) = a * e^{kt} + c

    if linear_method is None:
        linear_method = False

    if c0 is None:
        c0 = 0

    if linear_method:
        a, k = fit_exp_linear(t, y, c0)
        c = c0
        y_hat = model_func(t, a, k, c0)
    else:
        a, k, c = fit_exp_nonlinear(t, y, p0)
        y_hat = model_func(t, a, k, c)
    return a, k, c, y_hat


def model_func(t, a, k, c):
    return a * np.exp(k * t) + c


def fit_exp_linear(t, y, c=0):
    y = y - c
    y = np.log(y)
    k, a_log = np.polyfit(t, y, 1)
    a = np.exp(a_log)
    return a, k


def fit_exp_nonlinear(t, y, p0):
    opt_parms, parm_cov = sp.optimize.curve_fit(model_func, t, y, p0=p0, maxfev=200000)
    a, k, c = opt_parms
    return a, k, c