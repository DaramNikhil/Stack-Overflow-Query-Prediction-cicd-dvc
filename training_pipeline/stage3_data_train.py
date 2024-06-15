import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from utils import *
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics


def data_trainings(data_yaml):
    try:
        config = read_yaml(data_yaml)
        artifacts = config["artifacts"]
        artifacts_dir = os.path.join(artifacts["ARTIFACTS_DIR"])
        scaled_model_data_path = os.path.join(artifacts_dir, artifacts["MODEL_FEATURES"])
        train_data_path = os.path.join(scaled_model_data_path, artifacts["FEATURE_TRAIN_DATA"])
        model_dir = os.path.join(artifacts_dir, artifacts["MODEL_DIR"])
        os.makedirs(model_dir, exist_ok=True)
        model_save_path = os.path.join(model_dir, artifacts["MODEL_FILE"])
        train_data_matrix = joblib.load(train_data_path)
        X = train_data_matrix[:, 2:]
        label = np.squeeze(train_data_matrix[:,1].toarray())
        model = RandomForestClassifier(n_estimators=120, min_samples_split=16, random_state=2021, n_jobs=-1)
        model.fit(X, label)
        joblib.dump(model, model_save_path)

    except Exception as e:
        raise e

        