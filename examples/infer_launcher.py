import json
from mom_trans.new_inference import run_online_learning
from settings.fixed_params import MODLE_PARAMS
from settings.default import BINANCE_SYMBOLS

def load_best_hp(path):
    with open(path, "r") as f:
        best_hp = json.load(f)
    
    return best_hp
    

ASSET_CLASS_MAPPING = dict(zip(BINANCE_SYMBOLS, ["COMB"] * len(BINANCE_SYMBOLS)))
ONLINE_EXPERIMENT      = "experiment_binance_100assets_tft_cpnone_len63_notime_div_v2"
TEST_MODE = False
TRAIN_VALID_RATIO = 0.90
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True

params = MODLE_PARAMS.copy()
params["total_time_steps"] = 252
params["architecture"] = "TFT"
params["evaluate_diversified_val_sharpe"] = EVALUATE_DIVERSIFIED_VAL_SHARPE
params["train_valid_ratio"] = TRAIN_VALID_RATIO
params["time_features"] = TIME_FEATURES
params["force_output_sharpe_length"] = FORCE_OUTPUT_SHARPE_LENGTH
params["online_learning"] = True
params["continual_training"] = True
best_hp = load_best_hp("results/experiment_binance_100assets_tft_cpnone_len63_notime_div_v2/hyperparameters.json")

params.update(best_hp)

run_online_learning(
    experiment_name        = ONLINE_EXPERIMENT,
    features_file_path     = "./data/binance_cpd_nonelbw.csv",
    params                 = params,
    window_size            = 8064,   # 12주
    delta                  = 2016,   # 3주
    hp_minibatch_size = 64,
    asset_class_dictionary = ASSET_CLASS_MAPPING,
    fine_tune_epochs       = 1
)