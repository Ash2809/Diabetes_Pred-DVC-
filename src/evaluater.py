import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow
from urllib.parse import urlparse


os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Ash2809/my-first-repo.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Ash2809"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "35daa4e00a839552eb11a8de775274f491754fd7"

def evaluate(data_path,model_path):
    data=pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/Ash2809/my-first-repo.mlflow")

    model=pickle.load(open(model_path,'rb'))

    predictions=model.predict(X)
    accuracy=accuracy_score(y,predictions)

    mlflow.log_metric("accuracy",accuracy)
    print("Model accuracy:{accuracy}")

if __name__=="__main__":
    params = yaml.safe_load(open("params.yaml"))
    evaluate(params['trainer']['data'], params['trainer']['model'])