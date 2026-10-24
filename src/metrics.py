"""Error metrics."""
import numpy as np


def rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mae(y, yhat):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def max_abs(y, yhat):
    return float(np.max(np.abs(np.asarray(y) - np.asarray(yhat))))


def r2(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    ss = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else float("nan")


def rel_err(est, true):
    return float((est - true) / true)


def summary(y, yhat, prefix=""):
    return {f"{prefix}rmse": rmse(y, yhat), f"{prefix}mae": mae(y, yhat), f"{prefix}max_abs": max_abs(y, yhat)}
