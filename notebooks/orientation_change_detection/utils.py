
import numpy as np
import pandas as pd

def get_day_orientations(cfg):
    # creating randomly chosen orientations that stay constant during each experimental "day"
    n_days = cfg.experiment.n_days

    np.random.seed(42)
    day_orientations = np.random.rand(n_days, 2) * np.pi
    return day_orientations


def add_correct(df):
    df["correct"] = df["target"] == df["target_pred"]
    df["correct"].mean()
    return df


def add_change_bins(df):
    change_bins = np.logspace(0, np.log10(100), num=8) # log10 bins
    df["change_bin"] = pd.cut(np.abs(df["change"]), bins=change_bins, include_lowest=True,
                              labels=[(change_bins[i]+change_bins[i+1])/2 for i in range(len(change_bins)-1)])
    return df