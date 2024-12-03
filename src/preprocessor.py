import pandas as pd
import sys
import yaml 
import os

params = yaml.safe_load(open("C:\MLOPS\Diabetes_Pred-DVC-\params.yaml"))

def preprocess(input, output):
    data = pd.read_csv(input)

    os.makedirs(os.path.dirname(output), exist_ok = True)
    data.to_csv(output, header = None, index = False)
    print(f"Preprocessed data saved to {output}")

if __name__ == "__main__":
    preprocess(params['preprocessor']['input'], params['preprocessor']['output'])
