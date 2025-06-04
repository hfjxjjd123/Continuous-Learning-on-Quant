from mom_trans.new_inference import run_online_learning
from settings.fixed_params import MODLE_PARAMS
from some_module import load_best_hp   # ← 사용자가 마련한 함수

params = MODLE_PARAMS.copy()
best_hp = load_best_hp("path/to/best_hyperparameters.json")

run_online_learning(
    experiment_name        = "experiment_binance_..._v1",
    features_file_path     = "./data/binance_cpd_nonelbw.csv",
    params                 = params,
    best_hyperparameters   = best_hp,
    window_size            = 8064,   # 12주
    delta                  = 2016,   # 3주
    fine_tune_epochs       = 1,
)