from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn
import sys, os
sys.path.append(os.getcwd())
from utils import *
from prediction_pipeline import prediction_pipeline
import pandas as pd
import numpy as np


app = FastAPI()


@app.get("/")
def predict_page():
    message = "<h1> Wellcome to the prediction api</h1>"
    message2 = "<h2> click the below link to get predictions</h2>"
    link = "<a href='http://127.0.0.1'> Predict for inferencing batch data</a>"
    msg = message+message2+link
    return HTMLResponse(content=msg)


@app.get('/predict')
def prediction_batch_data():
    load_model = "artifacts/models/model.pkl"
    transformer_model = "artifacts/model_features/pipeline.pkl"
    pred_data_file = "data.tsv"
    prediction_pipeline.prediction_for_batch(load_model, transformer_model, pred_data_file)
    prediction_data = "prediction.csv"
    df = pd.read_csv(prediction_data)
    return Response(content=df.to_csv(index=False), media_type="text/csv")


if __name__ == '__main__':
    uvicorn.run(host="127.0.0.1", app=app)
