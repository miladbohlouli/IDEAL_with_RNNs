# Todo:
#   1. Try to come up with an adaptive resampling method

from matplotlib import pyplot as plt
import torch
import numpy as np
from matplotlib.dates import DateFormatter


def visulaize(dates, target, predicted, figure_size=[15, 10], date_format=None, y_lim = None):
    prediction_length = len(predicted)
    target /= 10
    predicted /= 10
    if y_lim is None:
        # Figuring put the limits
        max_val = np.ceil(max([target.max(), predicted.max()]))
        min_val = np.floor(min([target.min(), predicted.min()]))
        y_lim = [min_val, max_val]

    fig, ax = plt.subplots(figsize=figure_size)
    ax.set(
        xlabel="date(Weekday dd.mm)",
        ylabel="C",
        ylim=y_lim
    )
    ax.set_title("Visualization of the temperature data collected from households", fontsize=20)
    if date_format is None: date_format = "%a %H.%M"

    ax.plot(
        dates,
        target
    )
    ax.plot(
        dates[-prediction_length:],
        predicted
    )
    date_form = DateFormatter(date_format)
    ax.xaxis.set_major_formatter(date_form)
    ax.legend(["target", "predicted"])
    return fig


def custom_collate(batch):
    data = torch.as_tensor(np.array([b[0] for b in batch]))
    dates = np.array([b[1] for b in batch])
    return data, dates


def MSE_loss(ground_truth, results):
    return (ground_truth - results).pow(2).sum(axis=1).mean()
