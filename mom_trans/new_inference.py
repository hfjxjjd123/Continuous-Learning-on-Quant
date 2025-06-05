from pathlib import Path
import os
import math
import pandas as pd
import datetime as dt
import json
from mom_trans.classical_strategies import calc_performance_metrics
from typing import Tuple, List, Dict
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel
from mom_trans.online_inputs import ModelFeatures

def _get_directory_name(
    experiment_name: str
) -> str:
    return os.path.join("results", experiment_name)

def _load_csv(path: str, time_col: str = "Time") -> pd.DataFrame:
    # Time 컬럼에서 Timezone을 제거하고, Unix timestamp (초 단위)로 변환

    df = pd.read_csv(path, index_col=0)
    df.index = pd.to_datetime(df.index)
    
    if "Time" not in df.columns:
        print("NO TIME INSIDE")
        df['Time'] = df.index
    return df

# ======================================================================
#                      ONLINE‑LEARNING HELPER
# ======================================================================
def run_online_learning(
    experiment_name: str,
    features_file_path: str,
    params: dict,
    window_size: int,
    delta: int,
    fine_tune_epochs: int = 1,
    hp_minibatch_size: int = 64,
    asset_class_dictionary = Dict[str, str]
):
    """
    Continues training a *pre‑trained* DeepMomentumNetwork model in an
    online fashion.

    The function:
    1. Loads **one** model from `params`.
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
    params / params
        Same dictionaries you would pass to `TftDeepMomentumNetworkModel`.
        These are **not** modified.
    window_size : int
        Number of rows (time steps) inside each online window.
    delta : int
        Shift (in rows) between consecutive windows.
    fine_tune_epochs : int, default 1
        Extra epochs to train on each new window.
    hp_minibatch_size : int | None
        If given, use this batch‑size instead of the value inside
        `params`.
    asset_class_dictionary : dict[str, str] | None
        Needed only if you want to post‑process metrics per asset class.
    """
    
    output_dir = Path("results") / experiment_name

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
    print(f"n_windows: {n_windows}")

    # ------------------------------------------------------------------
    # 3) Iterate over sliding windows
    # ------------------------------------------------------------------
    aggregate_results = []
    for w in range(n_windows):
        start = w * delta
        end   = start + window_size
        train_window = full_df.iloc[start:end].copy()
        test_window = full_df.iloc[end: end+delta].copy()

        # ----------------------------------------------------------------
        # Build ModelFeatures for *one* window
        # ----------------------------------------------------------------
        #TODO checkpoint2
        features = ModelFeatures(
            train_window,
            test_window,
            total_time_steps = params["total_time_steps"],
            changepoint_lbws=None,
            train_valid_sliding=False,
            transform_real_inputs = False,
            train_valid_ratio=0.9,
            split_tickers_individually=True,
            add_ticker_as_static=False,
            time_features=False,
            lags=None,
            asset_class_dictionary=None,
            static_ticker_type_feature = False,
        )
        print(f"HOPE PASS HERE")

        #TODO 검증필요
        # We treat the same data twice:
        train_data  = features.train
        test_data   = features.test_fixed     # identical slice

        # --------------------------------------------------------------
        # 3a) Continue training
        # --------------------------------------------------------------
        dmn = TftDeepMomentumNetworkModel(
            project_name     = experiment_name,
            hp_directory        = output_dir / "hp",
            hp_minibatch_size   = [params["batch_size"]],
            **params,
            **features.input_params,
            **{
                "column_definition": features.get_column_definition(),
                "num_encoder_steps": 0,  # TODO artefact
                "stack_size": 1,
                "num_heads": 4,  # TODO to fixed params
            },
        )
        model = dmn.load_model(params)
        if hp_minibatch_size is not None:
            params["batch_size"] = hp_minibatch_size

        print(f"#21 PASS?")
    
        #TODO checkpoint -> need to fix
        model.fit(
            *ModelFeatures._unpack(train_data)[:3],                 # data, labels, weights
            epochs      = fine_tune_epochs,
            batch_size  = params["batch_size"],
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

        # --------------------------------------------------------------
        # 3c) Store per‑window results & metrics
        # --------------------------------------------------------------
        # 3c‑1) save raw captured returns of this window
        out_csv = output_dir / f"captured_returns_window{w:04d}.csv"
        res_df.to_csv(out_csv, index=False)

        # 3c‑2) calculate & save performance metrics for this window
        metrics = calc_performance_metrics(
            res_df.set_index("time"),
            metric_suffix=f"_window{w:04d}",
            num_identifiers=features.num_tickers,
        )
        out_json = output_dir / f"metrics_window{w:04d}.json"
        with open(out_json, "w") as fp:
            json.dump(metrics, fp, indent=2, default=float)

        model.save_weights(output_dir / f"ckpt_window{w:04d}.weights.h5")

    # ------------------------------------------------------------------
    # 4) Save concatenated results to disk
    # ------------------------------------------------------------------
    all_results = pd.concat(aggregate_results, ignore_index=True)
    all_results.to_csv(output_dir / "captured_returns_online.csv", index=False)
    print(f"[online‑learning] Finished. Results saved to "
          f"{output_dir}/captured_returns_online.csv")