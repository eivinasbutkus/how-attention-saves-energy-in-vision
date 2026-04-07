
from scipy.optimize import curve_fit
import numpy as np
import pandas as pd

from .paths import RESULTS_DIR


def get_results_dir(cfg):
    results_dir = RESULTS_DIR / cfg.experiment.name
    results_dir.mkdir(exist_ok=True, parents=True)
    return results_dir

def get_seaborn_colors(cfg):
    plot_cfg = cfg.plotting
    models = list(plot_cfg.model_names.keys())
    model_names = list(plot_cfg.model_names.values())

    seaborn_colors = {
        "hue" : "model",
        "hue_order" : models,
        "palette" : [dict(plot_cfg.model_colors)[name] for name in models],
    }
    return  models, model_names, seaborn_colors

def sigmoid(x, L, x0, k, b):
    """
    L: the curve's maximum value
    x0: the x-value of the sigmoid's midpoint
    k: the steepness/growth rate
    b: the y-offset
    """
    return L / (1 + np.exp(-k * (x - x0))) + b


# Define Richards curve (generalized logistic)
def richards(x, L, x0, k, nu, b):
    """
    Richards curve (generalized logistic) - most flexible asymmetric sigmoid
    
    When nu=1, this reduces to the standard logistic/sigmoid function.
    nu < 1: slower approach to upper asymptote (similar to Gompertz)
    nu > 1: slower approach to lower asymptote
    
    L: the curve's maximum value (amplitude)
    x0: the x-value of the inflection point
    k: the growth rate
    nu: asymmetry parameter (controls which asymptote is approached more slowly)
    b: the y-offset (lower asymptote)
    """
    return L / (1 + np.exp(-k * (x - x0)))**(1/nu) + b


# to visualize energy-accuracy trade-off
def fit_sigmoid(X, y, X_pred):
    p0 = [max(y) - min(y), np.median(X), 1, min(y)]
    bounds = (
        [0, min(X), 0, min(y)], 
        [max(y) - min(y) + 1, max(X), 20, max(y)]
    )

    try:
        # Fit sigmoid
        popt, pcov = curve_fit(sigmoid, X, y, p0=p0, maxfev=10000, bounds=bounds)
        y_pred = sigmoid(X_pred, *popt)
        return y_pred
        
    except RuntimeError as e:
        print(f"Sigmoid fitting error: {e}")
        # Fall back to simpler method if fitting fails


def fit_sigmoid_with_ci(X, y, X_pred, n_bootstrap=1000, ci=0.95):
    """
    Fit sigmoid with confidence intervals using bootstrap
    
    Parameters:
    -----------
    X : array-like
        Input x values
    y : array-like
        Target y values
    X_pred : array-like
        X values for prediction
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval (0-1)
    
    Returns:
    --------
    y_pred : array-like
        Predicted y values
    y_lower : array-like
        Lower bound of confidence interval
    y_upper : array-like
        Upper bound of confidence interval
    popt : array-like
        Optimal parameters from original fit
    """
    # Initial fitting
    p0 = [max(y) - min(y), np.median(X), 1, min(y)]
    bounds = (
        [0, min(X), 0, min(y)], 
        [max(y) - min(y) + 1, max(X), 20, max(y)]
    )
    
    try:
        # Fit sigmoid to original data
        popt, pcov = curve_fit(sigmoid, X, y, p0=p0, maxfev=10000, bounds=bounds)
        y_pred = sigmoid(X_pred, *popt)
        
        # Bootstrap for confidence intervals
        predictions = []
        indices = np.arange(len(X))
        
        for i in range(n_bootstrap):
            print(f"Bootstrap iteration {i+1}/{n_bootstrap}", end="\r")
            # Resample with replacement
            resample_idx = np.random.choice(indices, size=len(indices), replace=True)
            X_resample = X[resample_idx]
            y_resample = y[resample_idx]
            
            try:
                # Fit sigmoid to resampled data
                popt_boot, _ = curve_fit(
                    sigmoid, X_resample, y_resample, 
                    p0=p0, maxfev=5000, bounds=bounds
                )
                # Predict using bootstrap parameters
                y_boot = sigmoid(X_pred, *popt_boot)
                predictions.append(y_boot)
            except RuntimeError:
                # Skip failed fits
                continue
        
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate confidence intervals at each X_pred point
        alpha = (1 - ci) / 2
        y_lower = np.percentile(predictions, 100 * alpha, axis=0)
        y_upper = np.percentile(predictions, 100 * (1 - alpha), axis=0)
        
        return y_pred, y_lower, y_upper, popt
        
    except RuntimeError as e:
        print(f"Sigmoid fitting error: {e}")
        return None, None, None, None


def fit_richards_with_ci(X, y, X_pred, n_bootstrap=1000, ci=0.95):
    """
    Fit Richards curve (generalized logistic) with confidence intervals using bootstrap
    
    The Richards curve is the most flexible sigmoid with an asymmetry parameter (nu).
    When nu=1, it's identical to standard logistic. nu<1 gives slower upper asymptote
    approach (like Gompertz), nu>1 gives slower lower asymptote approach.
    
    Parameters:
    -----------
    X : array-like
        Input x values
    y : array-like
        Target y values
    X_pred : array-like
        X values for prediction
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval (0-1)
    
    Returns:
    --------
    y_pred : array-like
        Predicted y values
    y_lower : array-like
        Lower bound of confidence interval
    y_upper : array-like
        Upper bound of confidence interval
    popt : array-like
        Optimal parameters from original fit [L, x0, k, nu, b]
    """
    # Initial fitting with 5 parameters (including nu)
    p0 = [max(y) - min(y), np.median(X), 1, 0.5, min(y)]  # Added nu parameter (starting at 0.5)
    bounds = (
        [0, min(X), 0, 0.01, min(y)],  # nu bounded at 0.01 to avoid division by zero
        [max(y) - min(y) + 1, max(X), 20, 5, max(y)]  # nu can go up to 5
    )
    
    try:
        # Fit Richards curve to original data
        popt, pcov = curve_fit(richards, X, y, p0=p0, maxfev=10000, bounds=bounds)
        y_pred = richards(X_pred, *popt)
        
        # Bootstrap for confidence intervals
        predictions = []
        indices = np.arange(len(X))
        
        for i in range(n_bootstrap):
            print(f"Bootstrap iteration {i+1}/{n_bootstrap}", end="\r")
            # Resample with replacement
            resample_idx = np.random.choice(indices, size=len(indices), replace=True)
            X_resample = X[resample_idx]
            y_resample = y[resample_idx]
            
            try:
                # Fit Richards curve to resampled data
                popt_boot, _ = curve_fit(
                    richards, X_resample, y_resample, 
                    p0=p0, maxfev=5000, bounds=bounds
                )
                # Predict using bootstrap parameters
                y_boot = richards(X_pred, *popt_boot)
                predictions.append(y_boot)
            except RuntimeError:
                # Skip failed fits
                continue
        
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate confidence intervals at each X_pred point
        alpha = (1 - ci) / 2
        y_lower = np.percentile(predictions, 100 * alpha, axis=0)
        y_upper = np.percentile(predictions, 100 * (1 - alpha), axis=0)
        
        return y_pred, y_lower, y_upper, popt
        
    except RuntimeError as e:
        print(f"Richards curve fitting error: {e}")
        return None, None, None, None


from scipy.odr import ODR, Model, RealData

def sigmoid_odr_func(params, x):
    """Wrapper for ODR - just reorders parameters"""
    return sigmoid(x, *params)

def fit_sigmoid_odr_with_ci(X, y, X_err, y_err, X_pred, n_bootstrap=1000, ci=0.95):
    """
    Fit sigmoid with confidence intervals using ODR and bootstrap
    
    Parameters:
    -----------
    X : array-like
        Input x values (log energy)
    y : array-like
        Target y values (accuracy)
    X_err : array-like
        Standard errors in X
    y_err : array-like
        Standard errors in y
    X_pred : array-like
        X values for prediction
    n_bootstrap : int
        Number of bootstrap samples
    ci : float
        Confidence interval (0-1)
    
    Returns:
    --------
    y_pred : array-like
        Predicted y values
    y_lower : array-like
        Lower bound of confidence interval
    y_upper : array-like
        Upper bound of confidence interval
    popt : array-like
        Optimal parameters from original fit [L, x0, k, b]
    """
    # Initial parameter guess
    p0 = [max(y) - min(y), np.median(X), 1.0, min(y)]
    
    try:
        # Fit sigmoid to original data using ODR
        model = Model(sigmoid_odr_func)
        data = RealData(X, y, sx=X_err, sy=y_err)
        odr = ODR(data, model, beta0=p0)
        output = odr.run()
        
        popt = output.beta
        y_pred = sigmoid_odr_func(popt, X_pred)
        
        # Bootstrap for confidence intervals
        predictions = []
        n = len(X)
        
        for i in range(n_bootstrap):
            print(f"Bootstrap iteration {i+1}/{n_bootstrap}", end="\r")
            
            # Resample with replacement
            resample_idx = np.random.choice(n, size=n, replace=True)
            X_resample = X[resample_idx]
            y_resample = y[resample_idx]
            X_err_resample = X_err[resample_idx]
            y_err_resample = y_err[resample_idx]
            
            try:
                # Fit sigmoid to resampled data using ODR
                data_boot = RealData(X_resample, y_resample, 
                                    sx=X_err_resample, sy=y_err_resample)
                odr_boot = ODR(data_boot, model, beta0=p0)
                output_boot = odr_boot.run()
                
                # Predict using bootstrap parameters
                y_boot = sigmoid_odr_func(output_boot.beta, X_pred)
                predictions.append(y_boot)
            except:
                # Skip failed fits
                continue
        
        # Convert to numpy array
        predictions = np.array(predictions)
        
        # Calculate confidence intervals at each X_pred point
        alpha = (1 - ci) / 2
        y_lower = np.percentile(predictions, 100 * alpha, axis=0)
        y_upper = np.percentile(predictions, 100 * (1 - alpha), axis=0)
        
        return y_pred, y_lower, y_upper, popt
        
    except Exception as e:
        print(f"ODR sigmoid fitting error: {e}")
        return None, None, None, None


def add_noise_bins(df):
    df["noise_bin"] = pd.cut(df["noise"], bins=2)
    df["noise_bin"] = df["noise_bin"].apply(lambda x: x.mid)
    return df


def add_n_distractors_bins(df):
    df["n_distractors_bin"] = pd.cut(df["n_distractors"], bins=4, include_lowest=True)
    df["n_distractors_bin"] = df["n_distractors_bin"].apply(lambda x: x.mid) # 1-2, 3-4, 5-6, 7-8
    return df



def load_dataset_df(path):
    dataset_df = pd.read_csv(path)
    dataset_df = add_noise_bins(dataset_df)
    return dataset_df


def load_human_df(cfg, path):
    print("\nloading human data...")
    # loading human data
    human_df = pd.read_csv(path)
    human_df["where_correct"] = human_df["where_error"] < cfg.analysis.human.where_correct_threshold

    # print human accuracy
    print('what human accuracy', human_df["what_correct"].mean())
    print('where human accuracy', human_df["where_correct"].mean())

    # filter out the users that didn't see 389 images
    user_session_ids = human_df["user_session_id"].unique()
    n_images_per_user = human_df.groupby("user_session_id")["dataset_index"].nunique()
    print('n subjects before filtering those that didnt complete the study', human_df["user_session_id"].nunique())
    user_session_ids = n_images_per_user[n_images_per_user == 389].index
    human_df = human_df[human_df["user_session_id"].isin(user_session_ids)]
    print('n subjects after filtering those that didn\'t complete the study', human_df["user_session_id"].nunique())

    human_df = add_noise_bins(human_df) # add noise bins instead of using noise as a continuous variable
    human_df = add_n_distractors_bins(human_df) # add n_distractors bins instead of using n_distractors as a continuous variable

    # z-score difficulty judgement (for each participant)
    human_df["difficulty_z"] = human_df.groupby("user_session_id")["difficulty_response"].transform(lambda x: (x - x.mean()) / x.std())

    return human_df



def load_model_df(cfg, dataset_df):

    print("loading model data for", cfg.experiment.name)
    
    # loading raw model data
    model_df = pd.read_csv(get_results_dir(cfg) / f"model_{cfg.experiment.name}.csv")
    print(len(model_df), "rows in model dataframe before filtering")
    print(model_df["model"].unique())

    model_df = model_df[model_df["model"].isin(cfg.analysis.models)]

    # model_df = model_df.drop(columns="sample").groupby(["model", "energy_cost", "instance", "dataset_index", "t"]).mean(numeric_only=True).reset_index()
    # keeping just the first sample (samples are different inference runs on ths same image)
    # model_df = model_df[model_df["sample"] == 0]

    # add model_name
    model_df["model_name"] = model_df["model"].map(dict(cfg.plotting.model_names))

    # calculating model error and where correct
    where_xy_model_pred = np.array(model_df[["where_x_response", "where_y_response"]].values) # extracting model prediction
    # where_xy_model_pred += np.random.normal(0, constants.MODEL_MOTOR_NOISE, where_xy_model_pred.shape) # simulating motor noise in the model
    where_xy_gt = model_df[["where_x_gt", "where_y_gt"]].values # ground truth
    model_df["where_error"] = np.linalg.norm(where_xy_model_pred - where_xy_gt, axis=1) # calculating error
    model_df["where_correct"] = model_df["where_error"] < cfg.analysis.model.where_correct_threshold

    # summing energy used for action potentials and synaptic transmision
    model_df["energy"] = model_df["energy_ap"] + model_df["energy_st"]

    # add noise and n_distractors from dataset_df
    model_df = model_df.merge(dataset_df[["noise", "n_distractors"]], left_on="dataset_index", right_index=True)
    model_df["experiment"] = cfg.experiment.name

    model_df = add_noise_bins(model_df)
    model_df = add_n_distractors_bins(model_df)

    # normalize entropy by model x energy cost x instance
    model_df["what_entropy_z"] = model_df.groupby(["model", "energy_cost", "instance"])["what_entropy"].transform(lambda x: (x - x.mean()) / x.std())
    model_df["where_entropy_z"] = model_df.groupby(["model", "energy_cost", "instance"])["where_entropy"].transform(lambda x: (x - x.mean()) / x.std())
    model_df["what_where_entropy_z"] = (model_df["what_entropy_z"] + model_df["where_entropy_z"]) / 2 # average z-score from both measures
    model_df["energy_z"] = model_df.groupby(["model", "energy_cost", "instance"])["energy"].transform(lambda x: (x - x.mean()) / x.std())

    print(len(model_df), "rows in model dataframe when returning")

    return model_df




# model_df summary (last time step prediction, energy across all time steps)
def get_model_summary_df(model_df):
    print("\ngetting model summary dataframe...")

    factors = ["experiment", "model", "model_name", "energy_cost", "instance", "dataset_index", "sample"]
    
    # Get the last time step for each group
    last_step_df = model_df.loc[model_df.groupby(factors)["t"].idxmax()]
    
    # Compute the sum of energy for each group
    energy_sum_df = model_df.groupby(factors)["energy"].sum().reset_index()
    
    # Merge the last step accuracy and energy sum dataframes
    df = pd.merge(last_step_df[factors + ["what_correct", "where_correct", "where_error"]], 
                  energy_sum_df, on=factors)
    
    # average across dataset indices and samples
    factors.remove("dataset_index")
    factors.remove("sample")
    df = df.drop(columns=["dataset_index", "sample"]).groupby(factors).mean().reset_index()
    
    df.columns = factors + ["what_correct_last", "where_correct_last", "where_error_last", "energy"]
    
    df["log_energy"] = np.log(df["energy"])
    df["negative_energy"] = -df["log_energy"]
    df["energy_savings"] = df["log_energy"].max() - df["log_energy"]

    df["energy_cost_str"] = df["energy_cost"].apply(lambda x: "{:.10f}".format(x).rstrip("0"))
    
    return df



def get_image_df(model_df, human_df):
    """
    image df with model and human data where each row is an image
    """

    # MODEL
    factors = ["experiment", "model", "model_name", "energy_cost",  "instance", "dataset_index"]
    merge_factors = ["what_correct", "where_correct", "where_error", "what_entropy", "where_entropy", "what_entropy_z", "where_entropy_z"]
    
    # Get the last time step for each group (accuracy etc)
    last_step_df = model_df.loc[model_df.groupby(factors)["t"].idxmax()]

    # get the first time step to the the baseline energy from the no gain model
    first_step_df = model_df.loc[model_df.groupby(factors)["t"].idxmin()]
    first_step_df["baseline_energy"] = first_step_df["energy"]
    
    # Compute the sum of energy for each group
    energy_sum_df = model_df.groupby(factors)["energy"].sum().reset_index()
    
    # Merge the last step accuracy and energy sum dataframes
    df = pd.merge(last_step_df[factors + merge_factors],
                  energy_sum_df,
                  on=factors)
    
    df = pd.merge(df, first_step_df[factors + ["baseline_energy"]], on=factors)
    
    df.columns = factors + merge_factors + ["energy"] + ["baseline_energy"]

    df["log_energy"] = np.log(df["energy"])
    df["log_baseline_energy"] = np.log(df["baseline_energy"])
    df["energy_cost_str"] = df["energy_cost"].apply(lambda x: "{:.10f}".format(x).rstrip("0"))
    
    # HUMAN
    factors = ["dataset_index"]

    human_df = human_df.groupby(factors).agg(
        what_correct = ("what_correct", "mean"),
        where_correct = ("where_correct", "mean"),
        where_error = ("where_error", "mean"),
        difficulty_z = ("difficulty_z", "mean"),
        n_distractors = ("n_distractors", "first"),
        noise = ("noise", "first"),
    ).reset_index()

    # merge human and model data
    image_df = df.merge(human_df, on="dataset_index", suffixes=("_model", "_human"))

    return image_df



def get_energy_costs(cfg):
    e_min = cfg.experiment.energy_costs.min
    e_max = cfg.experiment.energy_costs.max
    steps = cfg.experiment.energy_costs.steps
    energy_costs = list(reversed(list(map(lambda x: x.round(2).item(), np.linspace(e_min, e_max, steps, dtype=float)))))
    return energy_costs