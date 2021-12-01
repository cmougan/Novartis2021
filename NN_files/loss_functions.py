import torch
import numpy as np


def interval_score_loss(predictions, real, alpha=0.28):
    """
    Taken from: https://stats.stackexchange.com/questions/194660/forecast-accuracy-metric-that-involves-prediction-intervals
    Need to predict lower and upper bounds of interval, use target to assess error.

    :param lower: Lower bound predictions
    :param upper: Upper bound predictions
    :param real: Target
    :param alpha: Alpha in metric in
    :return: Average of interval score loss
    """
    lower = predictions[:, 0]
    upper = predictions[:, 1]

    real_lower = 2 * torch.abs(real - lower) / alpha
    upper_real = 2 * torch.abs(upper - real) / alpha
    upper_lower = torch.abs(upper - lower)

    real_lower[real > lower] = 0
    upper_real[real < upper] = 0

    return torch.sum(real_lower + upper_real + upper_lower) / len(real)

'''Implementation of the quantile loss function'''

from typing import Sequence
import numpy as np


def quantile_loss(predictions: Sequence[float],
                  targets: Sequence[float],
                  quantile: float) -> float:
    '''Quantile loss function.

    Args:
        predictions (sequence of floats):
            Model predictions, of shape [n_samples,].
        targets (sequence of floats):
            Target values, of shape [n_samples,].
        quantile (float):
            The quantile we are seeking. Must be between 0 and 1.

    Returns:
        float: The quantile loss.
    '''
    # Convert inputs to NumPy arrays
    target_arr = np.asarray(targets)
    prediction_arr = np.asarray(predictions)

    # Compute the residuals
    res = target_arr - prediction_arr

    # Compute the mean quantile loss
    loss = np.mean(np.maximum(res, np.zeros_like(res)) * quantile +
                   np.maximum(-res, np.zeros_like(res)) * (1 - quantile))

    # Ensure that loss is of type float and return it
    return float(loss)


def smooth_quantile_loss(predictions: Sequence[float],
                         targets: Sequence[float],
                         quantile: float,
                         alpha: float = 0.4) -> float:
    '''The smooth quantile loss function from [1].

    Args:
        predictions (sequence of floats):
            Model predictions, of shape [n_samples,].
        targets (sequence of floats):
            Target values, of shape [n_samples,].
        quantile (float):
            The quantile we are seeking. Must be between 0 and 1.
        alpha (float, optional):
            Smoothing parameter. Defaults to 0.4.

    Returns:
        float: The smooth quantile loss.

    Sources:
        [1]: Songfeng Zheng (2011). Gradient Descent Algorithms for
             Quantile Regression With Smooth Approximation. International
             Journal of Machine Learning and Cybernetics.
    '''
    # Convert inputs to NumPy arrays
    target_arr = np.asarray(targets)
    prediction_arr = np.asarray(predictions)

    # Compute the residuals
    residuals = target_arr - prediction_arr

    # Compute the smoothened mean quantile loss
    loss = (quantile * residuals +
            alpha * np.log(1 + np.exp(-residuals / alpha)))

    # Ensure that loss is of type float and return it
    return float(loss.mean())
