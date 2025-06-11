from pathlib import Path
import os
import math
import pandas as pd
import datetime as dt
import json
from mom_trans.classical_strategies import calc_performance_metrics
from typing import Tuple, List, Dict
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel
 # Loss used when model is loaded without a compiled loss
from mom_trans.deep_momentum_network import SharpeLoss
from mom_trans.online_inputs import ModelFeatures
# --- Added for Elastic‑Weight‑Consolidation --------------------------

import tensorflow as tf

# ----------------------------------------------------------------------
# Custom loss builder: SharpeLoss + fixed EWC penalty (λ=100)
# ----------------------------------------------------------------------
def make_sharpe_ewc_loss(model,
                         theta_star: Dict[str, tf.Tensor],
                         fisher: Dict[str, tf.Tensor],
                         lambda_ewc: float = 100.0):
    """
    Builds a composite loss: SharpeLoss + λ · Σ_i F_i (θ_i − θ*_i)².

    Parameters
    ----------
    model : tf.keras.Model
        Current model whose weights are being updated.
    theta_star : dict[str, tf.Tensor]
        Snapshot of previous task parameters.
    fisher : dict[str, tf.Tensor]
        Diagonal Fisher Information estimates.
    lambda_ewc : float, default 100.0
        Regularisation strength.

    Returns
    -------
    Callable
        A Keras‑compatible loss(y_true, y_pred) function.
    """
    sharpe_loss_fn = SharpeLoss(output_size=model.output_shape[-1])

    # Pre‑cache valid (F_i, θ*_i) pairs whose shapes match the variable
    penalty_terms: list[tuple[tf.Variable, tf.Tensor, tf.Tensor]] = []
    for v in model.trainable_weights:
        if v.name in theta_star and v.name in fisher:
            if theta_star[v.name].shape == v.shape:
                penalty_terms.append((v, fisher[v.name], theta_star[v.name]))

    def loss(y_true, y_pred):
        base_loss = sharpe_loss_fn(y_true, y_pred)
        penalty = 0.0
        for v, Fi, theta_i in penalty_terms:
            penalty += tf.reduce_sum(Fi * tf.square(v - theta_i))
        return base_loss + lambda_ewc * penalty

    return loss


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

# ----------------------------------------------------------------------
#                         EWC HELPER FUNCTIONS
# ----------------------------------------------------------------------
def compute_fisher_information(model,
                               inputs,
                               labels,
                               batch_size: int = 64) -> Dict[str, tf.Tensor]:
    """
    Approximates the diagonal of the Fisher Information Matrix for EWC.
    For speed we use the squared gradients of the mini‑batch loss.

    Returns
    -------
    Dict[str, tf.Tensor]
        Mapping from variable name to Fisher diagonal estimates
        (same shape as variable tensor, on current device).
    """
    
    # Create an empty accumulator
    fisher = {v.name: tf.zeros_like(v, dtype=tf.float32)
              for v in model.trainable_weights}

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels)) \
                             .batch(batch_size, drop_remainder=False)

    n_batches = 0
    for x_b, y_b in dataset:
        n_batches += 1
        with tf.GradientTape() as tape:
            preds = model(x_b, training=False)
            sharpe_loss = SharpeLoss(output_size=preds.shape[-1])
            batch_loss = sharpe_loss(y_b, preds)
            batch_loss = tf.reduce_mean(batch_loss)

        grads = tape.gradient(batch_loss, model.trainable_weights)
        print(f"g length: {len(grads)}")
        for v, g in zip(model.trainable_weights, grads):
            if g is None:
                continue
            
            if fisher[v.name].shape != g.shape:
                continue
            
            fisher[v.name] += tf.square(g)

    # Average over #batches
    for k in fisher:
        fisher[k] /= float(max(n_batches, 1))
    return fisher

# def apply_ewc_penalty(model,
#                       theta_star: Dict[str, tf.Tensor],
#                       fisher: Dict[str, tf.Tensor],
#                       lambda_ewc: float = 100.0):
#     """
#     Adds EWC regularisation term  λ * Σ_i F_i (θ_i − θ*_i)^2  to `model.losses`.
#     Call *after* the model is built / weights loaded but *before*
#     `model.compile(...)`  or `model.fit(...)`.
#     """
#     print(f"len of v : {len(model.trainable_weights)}")
#     counter = 1
#     for v in model.trainable_weights:
#         if v.name not in theta_star:
#             counter += 1
#             continue
#         penalty = lambda_ewc * fisher[v.name] * tf.square(v - theta_star[v.name])
#         # Keras will add this term to the total loss automatically
#         model.add_loss(tf.reduce_sum(penalty))

# ----------------------------------------------------------------------
# Heuristic lambda estimation for EWC
# ----------------------------------------------------------------------
# def estimate_lambda_ewc(model,
#                         theta_star: Dict[str, tf.Tensor],
#                         fisher: Dict[str, tf.Tensor],
#                         inputs,
#                         labels,
#                         desired_ratio: float = 0.5,
#                         batch_size: int = 64) -> float:
#     """
#     Fast heuristic (히스테릭) method to choose λ_EWC.
#     Computes:
#         λ* ≈  desired_ratio ×  (L_orig / penalty_λ=1)
#     where
#         L_orig      = mean task‑loss on current window
#         penalty_λ=1 = Σ_i F_i (θ_i − θ*_i)^2  (with λ=1)

#     If either value is 0 → returns default 1e3.
#     """
#     # ---- 1) penalty value with λ=1 ---------------------------------
#     penalty_val = 0.0
#     for v in model.trainable_weights:
#         # must exist in previous Fisher AND snapshot dict
#         if v.name not in fisher or v.name not in theta_star:
#             continue

#         # skip if any shape mismatch
#         if (theta_star[v.name].shape != v.shape):
#             continue

#         penalty_val += tf.reduce_sum(
#             fisher[v.name] * tf.square(v - theta_star[v.name])
#         )
#     penalty_val = float(penalty_val.numpy())

#     # ---- 2) baseline task‑loss -------------------------------------
#     ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(batch_size)
#     loss_fn = model.loss
#     loss_sum = 0.0
#     n_batches = 0
#     for x_b, y_b in ds:
#         preds = model(x_b, training=False)
#         batch_loss = tf.reduce_mean(loss_fn(y_b, preds))
#         loss_sum += float(batch_loss.numpy())
#         n_batches += 1
#     L_orig = loss_sum / max(n_batches, 1)

#     if penalty_val == 0 or L_orig == 0:
#         return 1e3  # fallback

#     lambda_star = desired_ratio * (L_orig / penalty_val)
#     # Clamp to sensible positive range
#     return float(max(lambda_star, 1e-2))

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
    hp_minibatch_size = 64,
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

    Notes
    -----
    Supports EWC if you pass:
        "lambda_ewc": &lt;float&gt;            # fixed value
        "lambda_ewc": "auto"               # fast heuristic tuning
        "lambda_ewc_target_ratio": 0.5     # optional, default 0.5
    """

    # --------------------  EWC state  ---------------------------------
    # Fixed λ_EWC = 100  (no automatic tuning)
    lambda_ewc: float = 100.0
    auto_lambda = False
    ewc_fisher: Dict[str, tf.Tensor] = {}
    theta_star: Dict[str, tf.Tensor] = {}

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
    for w in range(9, n_windows):
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
            add_ticker_as_static=True,
            time_features=False,
            lags=None,
            asset_class_dictionary=asset_class_dictionary,
        )

        #TODO 검증필요
        # We treat the same data twice:
        train_data  = features.train
        test_data   = features.test_fixed     # identical slice

        # Explicitly unpack once here so objects exist for potential λ‑estimate
        train_inputs, train_labels, train_weights, _, _ = ModelFeatures._unpack(train_data)

        # --------------------------------------------------------------
        # 3a) Continue training
        # --------------------------------------------------------------
        dmn = TftDeepMomentumNetworkModel(
            project_name     = experiment_name,
            hp_directory        = output_dir / "2023-2025" / "hp",
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
        model = dmn.load_model(
            params,
            weights_path= output_dir/"2023-2025"/"best"/"checkpoints"/"checkpoint.weights.h5"
        )
        # --------------------------------------------------------------
        # (EWC) compile with Sharpe + λ·EWC penalty **if** previous task exists
        # --------------------------------------------------------------
        if theta_star:
            combined_loss = make_sharpe_ewc_loss(
                model,
                theta_star,
                ewc_fisher,
                lambda_ewc=lambda_ewc,   # fixed 100
            )
            model.compile(optimizer="adam", loss=combined_loss)
        else:
            model.compile(optimizer="adam", loss=SharpeLoss(output_size=params.get("output_size", 1)))
    
        #TODO checkpoint -> need to fix
        # print(f"hp size: {hp_minibatch_size}")
        # --------------------------------------------------------------
        # 3a) Continue training (online fine‑tuning)
        # --------------------------------------------------------------

        model.fit(
            x=train_inputs,
            y=train_labels,
            sample_weight=train_weights,
            epochs=fine_tune_epochs,
            batch_size=hp_minibatch_size,
            verbose=0,
            shuffle=False,  # keep temporal order for online learning
        )

        # --------------------------------------------------------------
        # (EWC)  Compute & accumulate Fisher after this window
        # --------------------------------------------------------------
        new_fisher = compute_fisher_information(
            model,
            train_inputs,
            train_labels,
            batch_size=hp_minibatch_size,
        )

        # Exponential moving‑average merge (α = 0.1)
        if not ewc_fisher:
            ewc_fisher = {k: tf.identity(v) for k, v in new_fisher.items()}
        else:
            for k in ewc_fisher:
                ewc_fisher[k] = 0.9 * ewc_fisher[k] + 0.1 * new_fisher[k]

        # Snapshot current parameters θ*
        theta_star = {v.name: tf.identity(v) for v in model.trainable_weights}

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

        model.save_weights(os.path.join(output_dir, "2023-2025", "best", "checkpoints", "checkpoint.weights.h5"))

    # ------------------------------------------------------------------
    # 4) Save concatenated results to disk
    # ------------------------------------------------------------------
    all_results = pd.concat(aggregate_results, ignore_index=True)
    all_results.to_csv(output_dir / "captured_returns_online.csv", index=False)
    print(f"[online‑learning] Finished. Results saved to "
          f"{output_dir}/captured_returns_online.csv")