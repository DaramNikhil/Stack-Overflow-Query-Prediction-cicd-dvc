import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import re
import random
import scipy.sparse as sparse
import numpy as np
import joblib


def process_data(f_in, target_tag, f_out_train, f_out_test, split):
    try:
        root = ET.fromstring(f_in.read())
        for child in root:
            pid = child.get("Id", "")
            label = 1 if target_tag in child.get("Tags", "") else 0
            title = re.sub(r"\s", " ", child.get("Title", "")).strip()
            body = re.sub(r"\s", " ", child.get("Body", "")).strip()
            combine_text = title + " " + body
            f_out = f_out_train if random.random() > split else f_out_test
            f_out.write(f"{pid}\t{label}\t{combine_text}\n")
    except Exception as e:
        raise e


def read_yaml(config_path):
    try:
        with open(config_path, "r") as data:
            config_data = yaml.safe_load(data)

        return config_data

    except Exception as e:
        raise e


def get_df(data_file, sep="\t"):
    df = pd.read_csv(
        data_file,
        encoding="utf-8",
        delimiter=sep,
        header=None,
        names=["Id", "Label", "Text"],
    )
    return df


def save_metrics(df, metrics, metrics_save_path):
    id_metrics = sparse.csr_matrix(df.Id.astype(np.int64)).T
    label_metrics = sparse.csr_matrix(df.Label.values.astype(np.int64)).T
    result = sparse.hstack([id_metrics, label_metrics, metrics], format="csr")
    joblib.dump(result, metrics_save_path)
