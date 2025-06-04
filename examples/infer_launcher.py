# example_infer_launcher.py
from mom_trans.new_inference import run_all_windows
from settings.fixed_params import MODLE_PARAMS
from settings.default import BINANCE_SYMBOLS


EXPERIMENT      = "experiment_binance_100assets_tft_cpnone_len63_notime_div_v1"
FEATURES2_CSV    = "datasets/online15m/BTCUSDT.csv"          # inference-용 feature 파일
FEATURES_CSV    = "./data/binance_cpd_nonelbw.csv"          # inference-용 feature 파일
ASSET_CLASS_MAPPING = dict(zip(BINANCE_SYMBOLS, ["COMB"] * len(BINANCE_SYMBOLS)))

TRAIN_INTERVALS = [(2018, 2023, 2025)]          # (start-year, test-start, test-end)

architecture = "TFT"
total_time_steps = 252
TEST_MODE = False
TRAIN_VALID_RATIO = 0.90
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True

params = MODLE_PARAMS.copy()
params.update({
    "total_time_steps": total_time_steps,
    "architecture": architecture,
    "evaluate_diversified_val_sharpe": EVALUATE_DIVERSIFIED_VAL_SHARPE,
    "train_valid_ratio": TRAIN_VALID_RATIO,
    "time_features": TIME_FEATURES,
    "force_output_sharpe_length": FORCE_OUTPUT_SHARPE_LENGTH,
})

CHANGEPOINT_LBWS = []   # CPD를 쓰지 않을 경우 빈 리스트

run_all_windows(
    experiment_name      = EXPERIMENT,
    features_file_path   = FEATURES_CSV,
    train_intervals      = TRAIN_INTERVALS,
    params               = params,
    changepoint_lbws     = CHANGEPOINT_LBWS,
    asset_class_dictionary = ASSET_CLASS_MAPPING,
    hp_minibatch_size    = [64],  # 아무거나; HP search를 건너뛰므로 영향 없음
    standard_window_size = 1      # 결과 요약에 쓰이는 표준 window 길이
)