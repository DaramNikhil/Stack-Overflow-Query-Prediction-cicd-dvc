from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
import uvicorn
import pandas as pd
import os
from prediction_pipeline import prediction_pipeline

app = FastAPI()

@app.get("/")
def predict_page():
    message = "<h1>Welcome to the prediction API</h1>"
    message2 = "<h2>Click the link below to get predictions</h2>"
    link = "<a href='/predict'> Predict for inferencing batch data</a>"
    msg = message + message2 + link
    return HTMLResponse(content=msg)

@app.get('/predict')
def prediction_batch_data():
    load_model = "artifacts/models/model.pkl"
    transformer_model = "artifacts/model_features/pipeline.pkl"
    pred_data_file = "data.tsv"
    prediction_pipeline.prediction_for_batch(load_model, transformer_model, pred_data_file)
    prediction_data = "prediction.csv"
    df = pd.read_csv(prediction_data)
    df_to_html = df.to_html(index=False)
    header = "<h1>Stackoverflow inferencing batch data</h1>"
    table_html = header + df_to_html
    return HTMLResponse(content=table_html)

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
