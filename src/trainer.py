import pandas as pd
import numpy as np

import pickle
import os
import sys
import yaml
import mlflow

from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/Ash2809/my-first-repo.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "Ash2809"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "35daa4e00a839552eb11a8de775274f491754fd7"

def hyperparametet_tuning(x_train, y_train, param_grid):
    rf=RandomForestClassifier()
    grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=3,n_jobs=-1,verbose=2)
    grid_search.fit(x_train,y_train)
    return grid_search

def train(data_path, model_path):
    data = pd.read_csv(data_path)
    x = data.drop(columns=['Outcome'], axis=1)
    y = data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/Ash2809/my-first-repo.mlflow")

    with mlflow.start_run():
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42,test_size = 0.2)
        signature = infer_signature(x_train,y_train)
        
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

        grid_search = hyperparametet_tuning(x_train, y_train, param_grid)
        best_model = grid_search.best_estimator_

        y_pred = best_model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy is:", accuracy)

        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_param("best_n_estimatios", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_sample_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_samples_leaf", grid_search.best_params_['min_samples_leaf'])

        cm = confusion_matrix(y_test, y_pred)
        cf = classification_report(y_test, y_pred)

        mlflow.log_text(str(cm),"confusion_matrix.txt")
        mlflow.log_text(cf, "classification_report.txt")

        tracking_url_type_store=urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store!='file':
            mlflow.sklearn.log_model(best_model,"model",registered_model_name="Best Model")
        else:
            mlflow.sklearn.log_model(best_model, "model",signature=signature)

        os.makedirs(os.path.dirname(model_path))

        filename=model_path
        pickle.dump(best_model,open(filename,'wb'))

        print(f"Model saved to {model_path}")

if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))
    train(params['trainer']['data'], params['trainer']['model'])