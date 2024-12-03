import pandas as pd
import sys
import yaml 
import os

params = yaml.safe_load(open("C:\MLOPS\Diabetes_Pred-DVC-\params.yaml"))

def preprocess(input):
    data = pd.read_csv(params['preprocessor']['input'])

