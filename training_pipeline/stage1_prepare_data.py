import os
import argparse
import shutil
from utils import process_data
from utils import read_yaml


def prepare_data(config_path):
    try:
        config = read_yaml(config_path)
        split_ratio = float(config["split_ratio"])
        input_data = config["local_data_file_path"]
        artifacts = config["artifacts"]
        prepare_data_dir = os.path.join(
            artifacts["ARTIFACTS_DIR"], artifacts["PREPARE_DATA"]
        )
        os.makedirs(prepare_data_dir, exist_ok=True)
        f_out_train = artifacts["TRAIN_DATA"]
        f_out_test = artifacts["TEST_DATA"]
        train_data_path = os.path.join(prepare_data_dir, f_out_train)
        test_data_path = os.path.join(prepare_data_dir, f_out_test)
        with open(input_data, "r", encoding="utf-8") as f_in:
            with open(train_data_path, "w", encoding="utf-8") as train_data:
                with open(test_data_path, "w", encoding="utf-8") as test_data:
                    process_data(
                        f_in=f_in,
                        target_tag="<python>",
                        f_out_train=train_data,
                        f_out_test=test_data,
                        split=split_ratio,
                    )
    except Exception as e:
        raise e
                