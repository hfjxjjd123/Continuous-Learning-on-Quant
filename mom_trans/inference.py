import os
import argparse
from settings.hp_grid import HP_MINIBATCH_SIZE
import pandas as pd
from settings.default import BINANCE_SYMBOLS
from settings.fixed_params import MODLE_PARAMS
# from mom_trans.backtest import run_all_windows
import numpy as np
from functools import reduce

# define the asset class of each ticker here - for this example we have not done this
TEST_MODE = False
ASSET_CLASS_MAPPING = dict(zip(BINANCE_SYMBOLS, ["COMB"] * len(BINANCE_SYMBOLS)))
TRAIN_VALID_RATIO = 0.90
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True
NAME = "experiment_binance_100assets"


def main(
    experiment: str,
    train_start: int,
    test_start: int,
    test_end: int,
    test_window_size: int,
    num_repeats: int,
):
    architecture = "TFT"
    lstm_time_steps = 252
    changepoint_lbws = None

    PROJECT_NAME = ""

        # TODO test_end += 1
        intervals = [
            (train_start, y, y + test_window_size)
            for y in range(test_start, test_end)
        ]

        params = MODLE_PARAMS.copy()
        params["total_time_steps"] = lstm_time_steps
        params["architecture"] = architecture
        params["evaluate_diversified_val_sharpe"] = EVALUATE_DIVERSIFIED_VAL_SHARPE
        params["train_valid_ratio"] = TRAIN_VALID_RATIO
        params["time_features"] = TIME_FEATURES
        params["force_output_sharpe_length"] = FORCE_OUTPUT_SHARPE_LENGTH
        params["online_learning"] = True
        params["continual_training"] = True

        if TEST_MODE:
            params["num_epochs"] = 1
            params["random_search_iterations"] = 2

        if changepoint_lbws:
            features_file_path = os.path.join(
                "data",
                f"binance_cpd_{np.max(changepoint_lbws)}lbw.csv",
            )
        else:
            features_file_path = os.path.join(
                "data",
                "binance_cpd_nonelbw.csv",
            )

        run_all_windows(
            PROJECT_NAME,
            features_file_path,
            intervals,
            params,
            changepoint_lbws,
            ASSET_CLASS_MAPPING,
            [32, 64, 128] if lstm_time_steps == 252 else HP_MINIBATCH_SIZE,
            test_window_size,
        )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(description="Run DMN experiment")
        parser.add_argument(
            "experiment",
            metavar="c",
            type=str,
            nargs="?",
            default="TFT-CPD-126-21",
            choices=[
                "LSTM",
                "LSTM-CPD-21",
                "LSTM-CPD-63",
                "TFT",
                "TFT-CPD-126-21",
                "TFT-SHORT",
                "TFT-SHORT-CPD-21",
                "TFT-SHORT-CPD-63",
                "btc_15m"
            ],
            help="Input folder for CPD outputs.",
        )
        parser.add_argument(
            "train_start",
            metavar="s",
            type=int,
            nargs="?",
            default=2018,
            help="Training start year",
        )
        parser.add_argument(
            "test_start",
            metavar="t",
            type=int,
            nargs="?",
            default=2022,
            help="Training end year and test start year.",
        )
        parser.add_argument(
            "test_end",
            metavar="e",
            type=int,
            nargs="?",
            default=2023,
            help="Testing end year.",
        )
        parser.add_argument(
            "test_window_size",
            metavar="w",
            type=int,
            nargs="?",
            default=1,
            help="Test window length in years.",
        )
        parser.add_argument(
            "num_repeats",
            metavar="r",
            type=int,
            nargs="?",
            default=1,
            help="Number of experiment repeats.",
        )

        args = parser.parse_known_args()[0]

        return (
            args.experiment,
            args.train_start,
            args.test_start,
            args.test_end,
            args.test_window_size,
            args.num_repeats,
        )

    main(*get_args())
