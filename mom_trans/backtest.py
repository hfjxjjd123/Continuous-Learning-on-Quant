import os
from typing import Tuple, List, Dict
import tensorflow as tf
import pandas as pd
import datetime as dt
import numpy as np
import shutil
import gc
import copy

import json

from mom_trans.model_inputs import ModelFeatures
from mom_trans.deep_momentum_network import LstmDeepMomentumNetworkModel
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel
from mom_trans.classical_strategies import (
    VOL_TARGET,
    calc_performance_metrics,
    calc_performance_metrics_subset,
    calc_sharpe_by_year,
    calc_net_returns,
)

from settings.default import BACKTEST_AVERAGE_BASIS_POINTS

from settings.hp_grid import HP_MINIBATCH_SIZE

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def _get_directory_name(
    experiment_name: str, train_interval: Tuple[int, int, int] = None
) -> str:
    """The directory name for saving results

    Args:
        experiment_name (str): name of experiment
        train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.

    Returns:
        str: folder name
    """
    if train_interval:
        return os.path.join(
            "results", experiment_name, f"{train_interval[1]}-{train_interval[2]}"
        )
    else:
        return os.path.join(
            "results",
            experiment_name,
        )


def _basis_point_suffix(basis_points: float = None) -> str:
    """Basis points suffix

    Args:
        basis_points (float, optional): bps valud. Defaults to None.

    Returns:
        str: suffix name
    """
    if not basis_points:
        return ""
    return "_" + str(basis_points).replace(".", "_") + "_bps"


def _interval_suffix(
    train_interval: Tuple[int, int, int], basis_points: float = None
) -> str:
    """Interval points suffix

    Args:
        train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year) Defaults to None.
        basis_points (float, optional): bps valud. Defaults to None.

    Returns:
        str: suffix name
    """
    return f"_{train_interval[1]}_{train_interval[2]}" + _basis_point_suffix(
        basis_points
    )


def _results_from_all_windows(
    experiment_name: str, train_intervals: List[Tuple[int, int, int]]
):
    """Save a json with results from all windows

    Args:
        experiment_name (str): experiment name
        train_intervals (List[Tuple[int, int, int]]): list of training intervals
    """
    return pd.concat(
        [
            pd.read_json(
                os.path.join(
                    _get_directory_name(experiment_name, interval), "results.json"
                ),
                # typ="series",
            )
            for interval in train_intervals
        ]
    )


def _get_asset_classes(asset_class_dictionary: Dict[str, str]):
    return np.unique(list(asset_class_dictionary.values())).tolist()


def _captured_returns_from_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    volatility_rescaling: bool = True,
    only_standard_windows: bool = True,
    volatilites_known: List[float] = None,
    filter_identifiers: List[str] = None,
    captured_returns_col: str = "captured_returns",
    standard_window_size: int = 1,
) -> pd.Series:
    """get sereis of captured returns from all intervals

    Args:
        experiment_name (str): name of experiment
        train_intervals (List[Tuple[int, int, int]]): list of training intervals
        volatility_rescaling (bool, optional): rescale to target annualised volatility. Defaults to True.
        only_standard_windows (bool, optional): only include full windows. Defaults to True.
        volatilites_known (List[float], optional): list of annualised volatities, if known. Defaults to None.
        filter_identifiers (List[str], optional): only run for specified tickers. Defaults to None.
        captured_returns_col (str, optional): column name of captured returns. Defaults to "captured_returns".
        standard_window_size (int, optional): number of years in standard window. Defaults to 1.

    Returns:
        pd.Series: series of captured returns
    """
    srs_list = []
    volatilites = volatilites_known if volatilites_known else []
    for interval in train_intervals:
        if only_standard_windows and (
            interval[2] - interval[1] == standard_window_size
        ):
            df = pd.read_csv(
                os.path.join(
                    _get_directory_name(experiment_name, interval),
                    "captured_returns_sw.csv",
                ),
            )

            if filter_identifiers:
                filter = pd.DataFrame({"identifier": filter_identifiers})
                df = df.merge(filter, on="identifier")
            num_identifiers = len(df["identifier"].unique())
            srs = df.groupby("time")[captured_returns_col].sum() / num_identifiers
            srs_list.append(srs)
            if volatility_rescaling and not volatilites_known:
                volatilites.append(annual_volatility(srs))
    if volatility_rescaling:
        return pd.concat(srs_list) * VOL_TARGET / np.mean(volatilites)
    else:
        return pd.concat(srs_list)


def save_results(
    results_sw: pd.DataFrame,
    output_directory: str,
    train_interval: Tuple[int, int, int],
    num_identifiers: int,
    asset_class_dictionary: Dict[str, str],
    extra_metrics: dict = {},
):
    """save results json

    Args:
        results_sw (pd.DataFrame): results dataframe
        output_directory (str): output directory
        train_interval (Tuple[int, int, int]): training interval
        num_identifiers (int): number of tickers
        asset_class_dictionary (Dict[str, str]): mapping of ticker to asset class
        extra_metrics (dict, optional): additional metrics to save. Defaults to {}.
    """
    asset_classes = ["ALL"]
    results_asset_class = [results_sw]
    print(f"DEBUG #6 - asset_class_dictionary: {asset_class_dictionary}")
    if asset_class_dictionary:
        results_sw["asset_class"] = results_sw["identifier"].map(
            lambda i: asset_class_dictionary[i]
        )
        classes = _get_asset_classes(asset_class_dictionary)
        for ac in classes:
            results_asset_class += [results_sw[results_sw["asset_class"] == ac]]
        asset_classes += classes

    print(f"DEBUG -- asset_classes => {asset_classes}")
    
    metrics = {}
    for ac, results_ac in zip(asset_classes, results_asset_class):
        suffix = _interval_suffix(train_interval)
        if ac == "ALL" and extra_metrics:
            ac_metrics = extra_metrics.copy()
        else:
            ac_metrics = {}
        for basis_points in BACKTEST_AVERAGE_BASIS_POINTS:
            suffix = _interval_suffix(train_interval, basis_points)
            if basis_points:
                results_ac_bps = results_ac.drop(columns="captured_returns").rename(
                    columns={
                        "captured_returns"
                        + _basis_point_suffix(basis_points): "captured_returns"
                    }
                )
            else:
                results_ac_bps = results_ac

            # TODO Empty DataFrame
            
            ac_metrics = {
                **ac_metrics,
                **calc_performance_metrics(
                    results_ac_bps.set_index("time"), suffix, num_identifiers
                ),
                **calc_sharpe_by_year(
                    results_ac_bps.set_index("time"), _basis_point_suffix(basis_points)
                ),
            }
        metrics = {**metrics, ac: ac_metrics}

    with open(os.path.join(output_directory, "results.json"), "w") as file:
        file.write(json.dumps(metrics, indent=4))

def aggregate_and_save_all_windows(
    experiment_name: str,
    train_intervals: List[Tuple[int, int, int]],
    asset_class_dictionary: Dict[str, str],
    standard_window_size: int,
):

    """Save a results summary, aggregating all windows

    Args:
        experiment_name (str): experiment name
        train_intervals (List[Tuple[int, int, int]]): list of (id, start_year, end_year)
        asset_class_dictionary (Dict[str, str]): map tickers to asset class
        standard_window_size (int): number of years in standard window
    """
    directory = _get_directory_name(experiment_name)
    os.makedirs(directory, exist_ok=True)

    # 모든 윈도우에서 생성된 results.json 누적
    all_results = _results_from_all_windows(experiment_name, train_intervals)

    # 기본 지표들
    _metrics = [
        "annual_return",
        "annual_volatility",
        "sharpe_ratio",
        "downside_risk",
        "sortino_ratio",
        "max_drawdown",
        "calmar_ratio",
        "perc_pos_return",
        "profit_loss_ratio",
    ]
    _rescaled_metrics = [
        "annual_return_rescaled",
        "annual_volatility_rescaled",
        "downside_risk_rescaled",
        "max_drawdown_rescaled",
    ]

    # 백테스트 BP suffix 적용된 전체 지표 리스트
    metrics = []
    rescaled_metrics = []
    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
        suf = _basis_point_suffix(bp)
        metrics += [m + suf for m in _metrics]
        rescaled_metrics += [m + suf for m in _rescaled_metrics]

    # 자산군 리스트 (ALL + 개별)
    if asset_class_dictionary:
        asset_classes = ["ALL"] + _get_asset_classes(asset_class_dictionary)
    else:
        asset_classes = ["ALL"]

    average_metrics: Dict[str, Dict[str, float]] = {}
    list_metrics: Dict[str, Dict[str, List[float]]] = {}

    # ticker→asset_class DataFrame
    asset_class_tickers = (
        pd.DataFrame.from_dict(asset_class_dictionary, orient="index", columns=["asset"])
        .reset_index()
        .rename(columns={"index": "symbol"})
        .set_index("asset")
    )

    # 각 asset_class별 집계
    print(f"DEBUG #7 : asset_classes 종류들 모두: {asset_classes}")
    for asset_class in asset_classes:
        # 빈 리스트로 초기화
        agg = {m: [] for m in metrics + rescaled_metrics}
        # 년도별 샤프 비율
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            agg[f"sharpe_ratio_years{_basis_point_suffix(bp)}"] = []

        # 해당 자산군의 results DataFrame
        df = all_results[asset_class]
        print(f"DEBUG #7 : df(results): {df}")
        print(f"DEBUG #7 : df index???: {df.index}")
        print(f"DEBUG #7 : asset_class: {asset_class}")
        

        # 윈도우별 통계 수집
        for interval in train_intervals:
            start_y, end_y = interval[1], interval[2]
            is_standard = (end_y - start_y) == standard_window_size

            # full window metrics
            if is_standard:
                for m in _metrics:
                    for bp in BACKTEST_AVERAGE_BASIS_POINTS:
                        col = m + _interval_suffix(interval, bp)
                        agg[m + _basis_point_suffix(bp)].append(df.at[col])

            # 각 년도별 샤프
            for bp in BACKTEST_AVERAGE_BASIS_POINTS:
                suf = _basis_point_suffix(bp)
                for y in range(start_y, end_y):
                    col = f"sharpe_ratio_{y}{suf}"
                    agg[f"sharpe_ratio_years{suf}"].append(df.at[col])

        # 캡처드 리턴스 기반 리스케일드 메트릭
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suf = _basis_point_suffix(bp)
            captured = _captured_returns_from_all_windows(
                experiment_name,
                train_intervals,
                volatility_rescaling=True,
                only_standard_windows=True,
                volatilites_known=agg["annual_volatility" + suf],
                filter_identifiers=(
                    None
                    if asset_class == "ALL"
                    else asset_class_tickers.loc[asset_class]["symbol"].tolist()
                ),
                captured_returns_col=f"captured_returns{suf}",
            )
            years = pd.to_datetime(captured.index).year
            for interval in train_intervals:
                if (interval[2] - interval[1]) == standard_window_size:
                    sub = captured[(years >= interval[1]) & (years < interval[2])]
                    res = calc_performance_metrics_subset(sub, f"_rescaled{suf}")
                    for m in _rescaled_metrics:
                        agg[m + suf].append(res[m + suf])

        # 평균 및 표준편차 계산
        history = copy.deepcopy(agg)
        stats = {k: float(np.mean(v)) for k, v in agg.items()}
        for bp in BACKTEST_AVERAGE_BASIS_POINTS:
            suf = _basis_point_suffix(bp)
            stats[f"sharpe_ratio_years_std{suf}"] = float(
                np.std(history[f"sharpe_ratio_years{suf}"])
            )

        average_metrics[asset_class] = stats
        list_metrics[asset_class] = {k: v for k, v in history.items()}

    # JSON으로 저장
    with open(os.path.join(directory, "average_results.json"), "w") as fp:
        json.dump(average_metrics, fp, indent=4)
    with open(os.path.join(directory, "list_results.json"), "w") as fp:
        json.dump(list_metrics, fp, indent=4)

    print(f">>> Aggregated results saved in {directory}")



def run_single_window(
    experiment_name: str,
    features_file_path: str,
    train_interval: Tuple[int, int, int],
    params: dict,
    changepoint_lbws: List[int],
    skip_if_completed: bool = True,
    asset_class_dictionary: Dict[str, str] = None,
    hp_minibatch_size: List[int] = HP_MINIBATCH_SIZE,
):
    """Backtest for a single test window

    Args:
        experiment_name (str): experiment name
        features_file_path (str): name of file, containing features
        train_interval (Tuple[int, int, int], optional): (start yr, end train yr / start test yr, end test year)
        params (dict): dmn experiment parameters
        changepoint_lbws (List[int]): CPD LBWs to be used
        skip_if_completed (bool, optional): skip, if previously completed. Defaults to True.
        asset_class_dictionary (Dict[str, str], optional): map tickers to asset class. Defaults to None.
        hp_minibatch_size (List[int], optional): minibatch size hyperparameter grid. Defaults to HP_MINIBATCH_SIZE.

    Raises:
        Exception: [description]
    """
    directory = _get_directory_name(experiment_name, train_interval)

    if skip_if_completed and os.path.exists(os.path.join(directory, "results.json")):
        print(
            f"Skipping {train_interval[1]}-{train_interval[2]} because already completed."
        )
        return

    raw_data = pd.read_csv(features_file_path, index_col=0, parse_dates=True)
    if "Time" not in raw_data.columns:
        print("NO TIME INSIDE")
        raw_data['Time'] = raw_data.index

    # Time 컬럼에서 Timezone을 제거하고, Unix timestamp (초 단위)로 변환
    raw_data["Time"] = raw_data["Time"].dt.tz_localize(None).astype('int64') // 10**9


    # TODO more/less than the one year test buffer
    model_features = ModelFeatures(
        raw_data,
        params["total_time_steps"],
        start_boundary=train_interval[0],
        test_boundary=train_interval[1],
        test_end=train_interval[2],
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=params["split_tickers_individually"],
        train_valid_ratio=params["train_valid_ratio"],
        add_ticker_as_static=(params["architecture"] == "TFT"),
        time_features=params["time_features"],
        lags=params["force_output_sharpe_length"],
        asset_class_dictionary=asset_class_dictionary,
    )

    hp_directory = os.path.join(directory, "hp")

    if params["architecture"] == "LSTM":
        dmn = LstmDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
        )
    elif params["architecture"] == "TFT":
        dmn = TftDeepMomentumNetworkModel(
            experiment_name,
            hp_directory,
            hp_minibatch_size,
            **params,
            **model_features.input_params,
            **{
                "column_definition": model_features.get_column_definition(),
                "num_encoder_steps": 0,  # TODO artefact
                "stack_size": 1,
                "num_heads": 4,  # TODO to fixed params
            },
        )
    else:
        dmn = None
        raise Exception(f"{params['architecture']} is not a valid architecture.")

    best_hp, best_model = dmn.hyperparameter_search(
        model_features.train, model_features.valid
    )
    val_loss = dmn.evaluate(model_features.valid, best_model)

    print(f"Best validation loss = {val_loss}")
    print(f"Best params:")
    for k in best_hp:
        print(f"{k} = {best_hp[k]}")

    with open(os.path.join(directory, "best_hyperparameters.json"), "w") as file:
        file.write(json.dumps(best_hp))

    # if predict_on_test_set:
    print("Predicting on test set...")

    results_sw, performance_sw = dmn.get_positions(
        model_features.test_sliding,
        best_model,
        sliding_window=True,
        years_geq=train_interval[1],
        years_lt=train_interval[2],
    )
    print(f"performance (sliding window) = {performance_sw}")

    raw_data = raw_data.drop(columns=["Time"])

    results_sw["time"] = results_sw["time"].dt.tz_localize("UTC")
    results_sw = results_sw.merge(
        raw_data.reset_index()[["symbol", "Time", "MVo"]].rename(
            columns={"symbol": "identifier", "Time": "time"}
        ),
        on=["identifier", "time"],
    )
    results_sw = calc_net_returns(
        results_sw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers
    )
    results_sw.to_csv(os.path.join(directory, "captured_returns_sw.csv"))

    # keep fixed window just in case
    results_fw, performance_fw = dmn.get_positions(
        model_features.test_fixed,
        best_model,
        sliding_window=False,
        years_geq=train_interval[1],
        years_lt=train_interval[2],
    )
    print(f"performance (fixed window) = {performance_fw}")
    results_fw = results_fw.merge(
        raw_data.reset_index()[["symbol", "Time", "MVo"]].rename(
            columns={"symbol": "identifier", "Time": "time"}
        ),
        on=["identifier", "time"],
    )
    results_fw = calc_net_returns(
        results_fw, BACKTEST_AVERAGE_BASIS_POINTS[1:], model_features.tickers
    )
    results_fw.to_csv(os.path.join(directory, "captured_returns_fw.csv"))

    with open(os.path.join(directory, "fixed_params.json"), "w") as file:
        file.write(
            json.dumps(
                dict(
                    **params,
                    **model_features.input_params,
                    **{
                        "changepoint_lbws": changepoint_lbws
                        if changepoint_lbws
                        else [],
                        "features_file_path": features_file_path,
                    },
                ),
                indent=4,
            )
        )

    # save model and get rid of the hp dir
    best_directory = os.path.join(directory, "best")
    best_model.save_weights(os.path.join(best_directory, "checkpoints", "checkpoint.weights.h5"))
    with open(os.path.join(best_directory, "hyperparameters.json"), "w") as file:
        file.write(json.dumps(best_hp, indent=4))

    save_results(
        results_sw,
        directory,
        train_interval,
        model_features.num_tickers,
        asset_class_dictionary,
        {
            "performance_sw": performance_sw,
            "performance_fw": performance_fw,
            "val_loss": val_loss,
        },
    )

    # get rid of everything and reset - TODO maybe not needed...
    del best_model
    gc.collect()
    tf.keras.backend.clear_session()
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


def run_all_windows(
    experiment_name: str,
    features_file_path: str,
    train_intervals: List[Tuple[int, int, int]],
    params: dict,
    changepoint_lbws: List[int],
    asset_class_dictionary=Dict[str, str],
    hp_minibatch_size=HP_MINIBATCH_SIZE,
    standard_window_size=1,
):
    print(f"RUN_ALL_WINDOWS \n {experiment_name} \n {features_file_path}")
    """Run experiment for multiple test intervals and aggregate results

    Args:
        experiment_name (str): experiment name
        features_file_path (str): name of file, containing features
        train_intervals (List[Tuple[int, int, int]]): klist of all training intervals
        params (dict): dmn experiment parameters
        changepoint_lbws (List[int]): CPD LBWs to be used
        asset_class_dictionary ([type], optional): map tickers to asset class. Defaults to None. Defaults to Dict[str, str].
        hp_minibatch_size ([type], optional): minibatch size hyperparameter grid. Defaults to HP_MINIBATCH_SIZE.
        standard_window_size (int, optional): standard number of years in test window. Defaults to 1.
    """
    # run the expanding window
    for interval in train_intervals:
        run_single_window(
            experiment_name,
            features_file_path,
            interval,
            params,
            changepoint_lbws,
            asset_class_dictionary=asset_class_dictionary,
            hp_minibatch_size=hp_minibatch_size,
        )

    aggregate_and_save_all_windows(
        experiment_name, train_intervals, asset_class_dictionary, standard_window_size
    )


def intermediate_momentum_position(w: float, returns_data: pd.DataFrame) -> pd.Series:
    """Position size for intermediate strategy.
    See https://arxiv.org/pdf/2105.13727.pdf

    Args:
        w (float): intermediate wweighting
        returns_data (pd.DataFrame): [description]

    Returns:
        pd.Series: series of position sizes
    """
    return w * np.sign(returns_data["norm_monthly_return"]) + (1 - w) * np.sign(
        returns_data["norm_annual_return"]
    )


def run_classical_methods(
    features_file_path,
    train_intervals,
    reference_experiment,
    long_only_experiment_name="long_only",
    tsmom_experiment_name="tsmom",
):
    """Run classical TSMOM method and Long Only as defined in https://arxiv.org/pdf/2105.13727.pdf.

    Args:
        features_file_path ([type]): file path containing the features.
        train_intervals ([type]): list of train/test intervalse
        reference_experiment ([type]): other experiment, testing against
        long_only_experiment_name (str, optional): name of long only experiment. Defaults to "long_only".
        tsmom_experiment_name (str, optional): name of TSMOM experiment. Defaults to "tsmom".
    """
    directory = _get_directory_name(long_only_experiment_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = _get_directory_name(tsmom_experiment_name)
    if not os.path.exists(directory):
        os.mkdir(directory)

    for train_interval in train_intervals:
        directory = _get_directory_name(tsmom_experiment_name, train_interval)
        if not os.path.exists(directory):
            os.mkdir(directory)
        raw_data = pd.read_csv(features_file_path, parse_dates=True)
        reference = pd.read_csv(
            f"results/{reference_experiment}/{train_interval[1]}-{train_interval[2]}/captured_returns_sw.csv",
            parse_dates=True,
        )
        returns_data = raw_data.merge(
            reference[["time", "identifier", "returns"]],
            left_on=["Time", "symbol"],
            right_on=["time", "identifier"],
        )
        returns_data["position"] = intermediate_momentum_position(0, returns_data)
        # returns_data["returns"] = returns_data["scaled_return_target"]
        returns_data["captured_returns"] = (
            returns_data["position"] * returns_data["returns"]
        )
        returns_data = returns_data.reset_index()[
            ["identifier", "time", "returns", "position", "captured_returns"]
        ]
        returns_data.to_csv(f"{directory}/captured_returns_sw.csv")

        directory = _get_directory_name(long_only_experiment_name, train_interval)
        if not os.path.exists(directory):
            os.mkdir(directory)
        returns_data["position"] = 1.0
        returns_data["captured_returns"] = (
            returns_data["position"] * returns_data["returns"]
        )
        returns_data.to_csv(f"{directory}/captured_returns_sw.csv")
