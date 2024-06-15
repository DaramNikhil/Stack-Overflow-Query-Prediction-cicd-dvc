import pandas as pd
import numpy as np
import sys, os
sys.path.append(os.getcwd())
from utils import *
import joblib
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import math
import json


def evaluation(data_yaml="config.yaml"):
    try:
        config = read_yaml(data_yaml)
        artifacts = config["artifacts"]
        artifacts_dir = os.path.join(artifacts["ARTIFACTS_DIR"])
        scaled_model_data_path = os.path.join(artifacts_dir, artifacts["MODEL_FEATURES"])
        test_data_path = os.path.join(scaled_model_data_path, artifacts["FEATURE_TEST_DATA"])
        model_dir = os.path.join(artifacts_dir, artifacts["MODEL_DIR"])
        model_save_path = os.path.join(model_dir, artifacts["MODEL_FILE"])
        test_data_matrix = joblib.load(test_data_path)
        model = joblib.load(model_save_path)
        X2 = test_data_matrix[:, 2:]
        label = np.squeeze(test_data_matrix[:,1].toarray())
        y_pred = model.predict(X2)
        prediction_by_class = model.predict_proba(X2)
        roc_score = metrics.roc_auc_score(label, prediction_by_class[:,1])
        avarage_precision_score = metrics.average_precision_score(label, prediction_by_class[:,1])
        model_score_dict = {"avarage_precision_score": avarage_precision_score,"roc_score":roc_score }
        model_score_json_path = artifacts["MODEL_SCORE"]
        json.dump(model_score_dict,open(model_score_json_path,"w"))
        precision, recall, theresold = metrics.precision_recall_curve(label, prediction_by_class[:,1])
        nth_point = math.ceil(len(precision)/100)
        prc_point = list(zip(precision, recall, theresold))[::nth_point]
        prec_curve_points = [{"precision":pr, "recall":re, "theresold":th} for pr,re, th in prc_point]
        precision_json_path = artifacts["PRECISION_JSON"]
        roc_json_path = artifacts["ROC_JSON"]
        json.dump(prec_curve_points, open(precision_json_path, "w"))
        fpr, rpr, tpr = metrics.roc_curve(label,prediction_by_class[:,1])
        roc_curve_points = {"roc":[{"precision":fpr, "recall":rpr, "theresold":tpr} for fpr,rpr, tpr in zip(fpr, rpr, tpr)]}
        json.dump(roc_curve_points, open(roc_json_path, "w"))
        print(roc_curve_points)

    except Exception as e:
        raise e