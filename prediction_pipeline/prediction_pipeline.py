import pandas as pd
import numpy as np
import joblib
import sys, os
sys.path.append(os.getcwd())
from utils import *
def single_query_prediction(model_path, transformer_path, query):
    try:
        model = joblib.load(model_path)
        transformer = joblib.load(transformer_path)
        data_query_transform = transformer.transform([query])
        y_pred = model.predict(data_query_transform)
        if y_pred == [1.0]:
            print("This query is related to Python")
        else:
            print("non related python query")

    except Exception as e:
        raise e



def prediction_for_batch(model_path, transformer_path, batch):
    try:
        model = joblib.load(model_path)
        transformer = joblib.load(transformer_path)
        test_data = get_df(batch)
        test_data_array = np.array(test_data["Text"].str.lower().values.astype("U"))
        metrix_test_data = transformer.transform(test_data_array)
        y_pred_arr = np.array(model.predict(metrix_test_data))
        test_data['prediction_data'] = y_pred_arr
        test_data['prediction_data'] = test_data["prediction_data"].replace({
                0.0:"non python query",
                1.0:"Query is related to python"
                
        })
        test_data.to_csv("prediction.csv", index=False)
        return y_pred_arr
    
    except Exception as e:
        raise e







        