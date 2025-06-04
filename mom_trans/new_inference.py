from pathlib import Path
import math
import pandas as pd
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel
from mom_trans.infer_inputs import ModelFeatures


def _load_csv(path: str, time_col: str = "Time") -> pd.DataFrame:
    """Utility – read CSV and set <time_col> as UTC DateTimeIndex."""
    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"`{time_col}` column missing in {path}")
    df[time_col] = pd.to_datetime(df[time_col], unit="s", utc=True)
    df = df.set_index(time_col).sort_index()
    return df

# ======================================================================
#                      ONLINE‑LEARNING HELPER
# ======================================================================
def run_online_learning(
    experiment_name: str,
    features_file_path: str,
    params: dict,
    best_hyperparameters: dict,
    window_size: int,
    delta: int,
    fine_tune_epochs: int = 1,
    batch_size_override: int | None = None,
    asset_class_dictionary: dict[str, str] | None = None,
    output_dir: str | None = None,
):
    """
    Continues training a *pre‑trained* DeepMomentumNetwork model in an
    online fashion.

    The function:
    1. Loads **one** model from `best_hyperparameters`.
    2. Slides through the dataset with `window_size` and `delta`.
    3. On each window it:
       - builds ModelFeatures,
       - *continues* training (`model.fit`) for `fine_tune_epochs`,
       - evaluates the window (Sharpe etc.),
       - saves the checkpoint & results.

    Parameters
    ----------
    experiment_name : str
        Folder name used by previous training run.  Checkpoints will be
        written under `results/{experiment_name}/online/`.
    features_file_path : str
        CSV with all data (must contain **Time** and **symbol** cols).
    params / best_hyperparameters
        Same dictionaries you would pass to `TftDeepMomentumNetworkModel`.
        These are **not** modified.
    window_size : int
        Number of rows (time steps) inside each online window.
    delta : int
        Shift (in rows) between consecutive windows.
    fine_tune_epochs : int, default 1
        Extra epochs to train on each new window.
    batch_size_override : int | None
        If given, use this batch‑size instead of the value inside
        `best_hyperparameters`.
    asset_class_dictionary : dict[str, str] | None
        Needed only if you want to post‑process metrics per asset class.
    output_dir : str | None
        Where to put csv results.  Defaults to
        `results/{experiment_name}/online/`.
    """

    if output_dir is None:
        output_dir = Path("results") / experiment_name / "online"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load the *full* dataframe once and sort by index
    # ------------------------------------------------------------------
    full_df = _load_csv(
        features_file_path,
        time_col="Time",
    ).sort_index()

    # Determine total number of steps and how many windows
    total_steps = len(full_df)
    n_windows = math.floor((total_steps - window_size) / delta) + 1
    print(f"[online‑learning] total steps={total_steps}, "
          f"window_size={window_size}, delta={delta}, "
          f" → {n_windows} windows")

    # ------------------------------------------------------------------
    # 2) Build model (weights will be updated in‑place)
    # ------------------------------------------------------------------
    dmn = TftDeepMomentumNetworkModel(
        experiment_name     = experiment_name + "_online",
        hp_directory        = output_dir / "hp",
        hp_minibatch_size   = [best_hyperparameters["batch_size"]],
        **params,
    )
    model = dmn.load_model(best_hyperparameters)
    if batch_size_override is not None:
        best_hyperparameters["batch_size"] = batch_size_override

    # ------------------------------------------------------------------
    # 3) Iterate over sliding windows
    # ------------------------------------------------------------------
    aggregate_results = []
    for w in range(n_windows):
        start = w * delta
        end   = start + window_size
        df_window = full_df.iloc[start:end].copy()

        # ----------------------------------------------------------------
        # Build ModelFeatures for *one* window
        # ----------------------------------------------------------------
        mf = ModelFeatures(
            df_window,
            test_fixed_ratio = 1.0,      # everything is “train” first
            test_start_year  = None,     # not used here
            test_end_year    = None,     # not used here
            time_features    = params.get("time_features", False),
            standard_window_size = 1,
        )

        # We treat the same data twice:
        train_data  = mf.train
        test_data   = mf.test_fixed     # identical slice

        # --------------------------------------------------------------
        # 3a) Continue training
        # --------------------------------------------------------------
        model.fit(
            *ModelFeatures._unpack(train_data)[:3],                 # data, labels, weights
            epochs      = fine_tune_epochs,
            batch_size  = best_hyperparameters["batch_size"],
            verbose     = 0,
        )

        # --------------------------------------------------------------
        # 3b) Evaluate (Sharpe on this window)
        # --------------------------------------------------------------
        res_df, sharpe = dmn.get_positions(
            test_data,
            model,
            sliding_window = False,
        )
        res_df["window_id"] = w
        aggregate_results.append(res_df)

        # Save model checkpoint after each window
        model.save_weights(output_dir / f"ckpt_window{w:04d}.weights.h5")

    # ------------------------------------------------------------------
    # 4) Save concatenated results to disk
    # ------------------------------------------------------------------
    all_results = pd.concat(aggregate_results, ignore_index=True)
    all_results.to_csv(output_dir / "captured_returns_online.csv", index=False)
    print(f"[online‑learning] Finished. Results saved to "
          f"{output_dir}/captured_returns_online.csv")