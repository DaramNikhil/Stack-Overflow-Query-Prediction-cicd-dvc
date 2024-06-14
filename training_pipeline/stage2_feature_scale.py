import yaml
import os, sys
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib
import numpy as np

sys.path.append(os.getcwd())
from utils import *


def prepare_data2(data_yaml):
    try:
        config = read_yaml(data_yaml)
        artifacts = config["artifacts"]
        artifacts_dir = os.path.join(artifacts["ARTIFACTS_DIR"])
        prepared_dir = os.path.join(artifacts_dir, artifacts["PREPARE_DATA"])
        scaled_model_data_path = os.path.join(
            artifacts_dir, artifacts["MODEL_FEATURES"]
        )
        train_data_path = os.path.join(prepared_dir, artifacts["TRAIN_DATA"])
        test_data_path = os.path.join(prepared_dir, artifacts["TEST_DATA"])
        os.makedirs(scaled_model_data_path, exist_ok=True)

        feature_train_data_path = os.path.join(
            scaled_model_data_path, artifacts["FEATURE_TRAIN_DATA"]
        )
        feature_test_data_path = os.path.join(
            scaled_model_data_path, artifacts["FEATURE_TEST_DATA"]
        )
        pipeline_save_data_path = os.path.join(
            scaled_model_data_path, artifacts["PIPELINE_DATA"]
        )
        train_df = get_df(data_file=train_data_path)
        max_features = config["featurization"]["max_features"]
        norm = config["featurization"]["norms"]
        train_words = np.array(train_df["Text"].str.lower().values.astype("U"))
        cv = CountVectorizer(
            stop_words="english", max_features=max_features, ngram_range=(1, norm)
        )
        tf_idf_metrics = TfidfTransformer(smooth_idf=False)
        pipeline = Pipeline(
            [("count_vectorizer", cv), ("tfidf_transformer", tf_idf_metrics)]
        )
        pipeline.fit(train_words)
        joblib.dump(pipeline, pipeline_save_data_path)  # Pipeline model saved path
        train_metrics = pipeline.transform(train_words)
        save_metrics(train_df, train_metrics, feature_train_data_path)

        test_df = get_df(data_file=test_data_path)
        test_words = np.array(test_df["Text"].str.lower().values.astype("U"))
        test_metrics = pipeline.transform(test_words)
        save_metrics(test_df, test_metrics, feature_test_data_path)

    except Exception as e:
        raise e
