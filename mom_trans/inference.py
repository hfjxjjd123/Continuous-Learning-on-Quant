# inference.py

import os
import json
import pandas as pd
import numpy as np
import datetime as dt
from settings.fixed_params import MODLE_PARAMS
from settings.default import BINANCE_SYMBOLS


from mom_trans.infer_inputs import ModelFeatures
from mom_trans.momentum_transformer import TftDeepMomentumNetworkModel

def load_best_hyperparameters(json_path):
    """하이퍼파라미터 JSON 파일을 읽어 dict로 반환."""
    with open(json_path, 'r') as f:
        return json.load(f)


def prepare_features(nth, hp, df_path, total_time_steps, start_year, test_start_year, test_end_year):
    """
    1) new.csv를 불러와서 DatetimeIndex를 UTC로 변환
    2) ModelFeatures를 이용해 모델이 입력으로 받을 형태로 변환
    """
    # CSV 읽을 때, 첫 번째 열이 시간(Unix timestamp)이면:
    idx_start = 8064*(nth-1)
    idx_end = 8064*nth
    df = pd.read_csv(df_path, index_col=0)
    print(f"#8: {df.index}")
    df = df[idx_start:idx_end]
    df.index = pd.to_datetime(df.index)

    # ModelFeatures 생성 → train/valid/test split을 거치지만, inference만 할 경우 test 부분만 필요
    feat = ModelFeatures(
        df,
        total_time_steps,
        start_boundary=start_year,
        test_boundary=test_start_year,
        test_end=test_end_year,
        changepoint_lbws=changepoint_lbws,
        split_tickers_individually=hp["split_tickers_individually"],
        train_valid_ratio=hp["train_valid_ratio"],
        add_ticker_as_static=(hp["architecture"] == "TFT"),
        time_features=hp["time_features"],
        lags=hp["force_output_sharpe_length"],
        asset_class_dictionary=ASSET_CLASS_MAPPING,
    )

    # inference용으로 입력해줄 데이터는 feat.test_sliding(또는 test_fixed) 안의 딕셔너리 형태
    return feat


def build_and_load_model(hp_dict, model_weights_path, features):
    """
    1) 하이퍼파라미터로 TF 모델 아키텍처를 빌드
    2) 저장된 weight(.weights.h5)를 로드
    """
    # hp_dict 안에는 'hidden_layer_size', 'dropout_rate', 'max_gradient_norm', 'learning_rate', 'batch_size' 등
    dmn = TftDeepMomentumNetworkModel(
        "experiment_binance_100assets_tft_cpnone_len63_notime_div_v1",
        HP_DIR,
        hp["batch_size"],
        **params,
        **features.input_params,
        **{
            "column_definition": features.get_column_definition(),
            "num_encoder_steps": 0,  # TODO artefact
            "stack_size": 1,
            "num_heads": 4,  # TODO to fixed params
        },
    )


    model = dmn.evaluate(hp)
    model.load_weights(model_weights_path)

    return dmn, model


def run_inference(dmn, model, features_obj):
    """
    모델을 이용해 inference를 수행하고, 결과를 DataFrame으로 반환.
    - features_obj.test_sliding: {'inputs':…, 'outputs':…, 'active_entries':…, 'identifier':…, 'date':…}
    """
    X_test = features_obj.test_sliding["inputs"]      # shape = (N, time_steps, input_size)
    id_test = features_obj.test_sliding["identifier"] # shape = (N, 1, 1) 혹은 (N,) 형태
    date_test = features_obj.test_sliding["date"]     # shape = (N, 1, 1) 혹은 (N,) 형태

    # 예측
    preds = model.predict(X_test, batch_size=hp["batch_size"], verbose=1)
    # preds shape = (N, 1) 혹은 (N, time_horizon, output_dim) 형태

    # date_test, id_test 을 1차원으로 펴기
    # 예: date_test[:, -1, 0] → 마지막 타임스텝의 ‘time’ 을 꺼내거나 적절히 reshape
    dates = date_test.reshape(-1)      # 필요시 flatten
    ids   = id_test.reshape(-1).astype(str)

    # 결과 DataFrame 생성
    results_df = pd.DataFrame({
        "identifier": ids,
        "forecast_time": pd.to_datetime(dates),  # 이미 datetime64[ns]여야 함
        "prediction": preds.reshape(-1),
    })

    return results_df

def run_inference_slice(model, X_slice, id_slice, date_slice, batch_size):
    """
    주어진 데이터 슬라이스에 대해 모델 추론을 수행하고 결과를 DataFrame으로 반환.
    """
    preds = model.predict(X_slice, batch_size=batch_size, verbose=0)
    dates = date_slice.reshape(-1)
    ids = id_slice.reshape(-1).astype(str)

    results_df = pd.DataFrame({
        "identifier": ids,
        "forecast_time": pd.to_datetime(dates),
        "prediction": preds.reshape(-1),
    })
    return results_df

if __name__ == "__main__":
    # =============================================================================
    #  설정 섹션
    # =============================================================================
    NEW_CSV_PATH        = "./data/binance_cpd_nonelbw.csv"
    HP_JSON_PATH        = "./results/experiment_binance_100assets_tft_cpnone_len63_notime_div_v1/2022-2023/hyperparameters.json"
    HP_DIR        = "./results/experiment_binance_100assets_tft_cpnone_len63_notime_div_v1/2022-2023/hp"
    MODEL_WEIGHTS_PATH  = "./results/experiment_binance_100assets_tft_cpnone_len63_notime_div_v1/2022-2023/best/checkpoints/checkpoint.weights.h5"
    ASSET_CLASS_MAPPING = dict(zip(BINANCE_SYMBOLS, ["COMB"] * len(BINANCE_SYMBOLS)))

    # ModelFeatures 경계값
    START_YEAR         = 2023
    TEST_BOUNDARY_YEAR = 2023
    TEST_END_YEAR      = 2025

    # 3주(window) 단위 길이 (days 기준)
    WINDOW_DAYS = 21  # 3주 = 21일

    # =============================================================================
    # 1) 하이퍼파라미터 로드
    # =============================================================================
    TRAIN_VALID_RATIO = 0.90
    TIME_FEATURES = False
    FORCE_OUTPUT_SHARPE_LENGTH = None
    EVALUATE_DIVERSIFIED_VAL_SHARPE = True
    NAME = "experiment_binance_100assets"
    architecture = "TFT"
    lstm_time_steps = 252
    changepoint_lbws = None
    
    hp = load_best_hyperparameters(HP_JSON_PATH)
    params = MODLE_PARAMS.copy()
    params["num_epochs"] = 1
    params["total_time_steps"] = lstm_time_steps
    params["architecture"] = architecture
    params["evaluate_diversified_val_sharpe"] = EVALUATE_DIVERSIFIED_VAL_SHARPE
    params["train_valid_ratio"] = TRAIN_VALID_RATIO
    params["time_features"] = TIME_FEATURES
    params["force_output_sharpe_length"] = FORCE_OUTPUT_SHARPE_LENGTH
    hp.update(params)

    # =============================================================================
    # 2) 입력 데이터 전처리 (ModelFeatures 생성)
    # =============================================================================
    ith = 1
    features = prepare_features(
        ith,
        hp,
        df_path=NEW_CSV_PATH,
        total_time_steps=hp["total_time_steps"],
        start_year=START_YEAR,
        test_start_year=TEST_BOUNDARY_YEAR,
        test_end_year=TEST_END_YEAR
    )

    # =============================================================================
    # 3) 모델 빌드 및 weight 로드
    # =============================================================================
    dmn, model = build_and_load_model(
        hp_dict=hp,
        model_weights_path=MODEL_WEIGHTS_PATH,
        features = features
    )

    # =============================================================================
    # 4) 전체 Test‐set 준비
    # =============================================================================
    test_dict      = features.test_sliding
    print(f"DEBUGGING: {test_dict}")
    X_test_full    = test_dict["inputs"]
    id_test_full   = test_dict["identifier"]
    date_test_full = test_dict["date"]

    flat_dates = date_test_full.reshape(-1)
    flat_ids   = id_test_full.reshape(-1).astype(str)
    flat_dates = pd.to_datetime(flat_dates)

    # =============================================================================
    # 5) 3주 단위로 반복하면서 inference
    # =============================================================================
    test_start = pd.Timestamp(f"{TEST_BOUNDARY_YEAR}-01-01", tz="UTC")
    test_end   = pd.Timestamp(f"{TEST_END_YEAR}-05-09 23:59:59", tz="UTC")

    window_starts = []
    current = test_start
    while current < test_end:
        window_starts.append(current)
        current = current + pd.Timedelta(days=WINDOW_DAYS)

    window_bounds = []
    for start in window_starts:
        end = start + pd.Timedelta(days=WINDOW_DAYS)
        if end > test_end:
            end = test_end
        window_bounds.append((start, end))

    for idx, (w_start, w_end) in enumerate(window_bounds, start=1):
        mask = (flat_dates >= w_start) & (flat_dates < w_end)
        if not mask.any():
            continue

        X_slice    = X_test_full[mask]
        id_slice   = id_test_full[mask]
        date_slice = date_test_full[mask]

        results_df = run_inference_slice(
            model      = model,
            X_slice    = X_slice,
            id_slice   = id_slice,
            date_slice = date_slice,
            batch_size = hp["batch_size"],
        )

        out_filename = f"inference_eval{idx}.csv"
        results_df.to_csv(out_filename, index=False)
        print(f"[Info] Window {idx}: {w_start.date()} ~ {w_end.date()} → {len(results_df)} rows saved to {out_filename}")

    print("모든 3주 단위 inference가 완료되었습니다.")