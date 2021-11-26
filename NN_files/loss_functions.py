import torch


def interval_score_loss(predictions, real, alpha=0.2
                        ):
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
