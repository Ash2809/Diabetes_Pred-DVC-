# Diabetes_Pred-DVC-
This project demonstrates the use of MLOps practices to train a machine learning model using `RandomForestClassifier` with hyperparameter tuning, model tracking, and versioning through [MLFlow](https://mlflow.org/). The pipeline is designed to integrate with cloud-based tracking services, such as [Dagshub](https://dagshub.com), for managing and storing models.

## **Table of Contents**
1. [Overview](#overview)
2. [Dependencies](#dependencies)
3. [Data](#data)
4. [Training and Hyperparameter Tuning](#training-and-hyperparameter-tuning)
5. [MLFlow Integration](#mlflow-integration)
6. [How to Run](#how-to-run)
7. [Model Evaluation](#model-evaluation)
8. [Future Work](#future-work)

---

## **Overview**

This repository contains an MLOps pipeline for training a machine learning model on a classification problem. The pipeline:

- Reads the dataset.
- Prepares the data (feature selection, target column extraction).
- Performs hyperparameter tuning using `GridSearchCV` for the `RandomForestClassifier`.
- Tracks the model training and hyperparameter search results using `MLFlow`.
- Logs metrics (accuracy, confusion matrix, classification report) and model details to the MLFlow tracking server.
- Saves the best model locally and optionally logs it to a registered model in MLFlow.

This project demonstrates the typical workflow of training, logging, and serving models in a production environment with MLOps practices.

---

## **Dependencies**

To run this project, you need the following Python packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `mlflow`
- `pyyaml`
- `pickle`

You can install the required packages by running:

```bash
pip install -r requirements.txt
```

You will also need to set up an MLFlow server (local or cloud-based), and the environment variables `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME`, and `MLFLOW_TRACKING_PASSWORD` need to be set for authentication.

---

## **Data**

The project uses a CSV dataset where the target column is named `Outcome`. The dataset can be anything where you want to predict a binary outcome, but for simplicity, a sample classification dataset (e.g., diabetes prediction) is used in this case.

### Example `params.yaml` for dataset path:

```yaml
trainer:
  data: 'path_to_your_data.csv'
  model: 'model_output_directory/model.pkl'
```

Replace `'path_to_your_data.csv'` with the actual path to your dataset.

---

## **Training and Hyperparameter Tuning**

1. **Data Splitting**: The dataset is split into training and testing sets using `train_test_split` from `sklearn`.
2. **Hyperparameter Tuning**: We use `GridSearchCV` to tune hyperparameters for the `RandomForestClassifier`. The grid search is performed over the following parameters:
   - `n_estimators`: Number of trees in the forest.
   - `max_depth`: Maximum depth of the tree.
   - `min_samples_split`: Minimum number of samples required to split an internal node.
   - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.

3. **Model Evaluation**: Once the best model is found, the model is evaluated on the test set, and accuracy is logged to MLFlow, along with the confusion matrix and classification report.

---

## **MLFlow Integration**

MLFlow is used to track the following during the training process:

- **Model Metrics**: Metrics like accuracy are logged.
- **Model Parameters**: Hyperparameters used in the grid search (e.g., `n_estimators`, `max_depth`).
- **Model Artifacts**: The trained model is saved both locally and registered in the MLFlow tracking server.

### Example Code to Log a Model with MLFlow:

```python
if tracking_url_type_store != 'file':
    mlflow.sklearn.log_model(best_model, "model", registered_model_name="Best Model")
else:
    mlflow.sklearn.log_model(best_model, "model", signature=signature)
```

### Example Code to Log Metrics and Parameters:

```python
mlflow.log_metric("Accuracy", accuracy)
mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
```

The model is also saved locally using `pickle`:

```python
pickle.dump(best_model, open(filename, 'wb'))
```

---

## **How to Run**

To run the model training pipeline:

1. Ensure that you have set the appropriate tracking URI, username, and password for MLFlow in your environment variables.
2. Configure the `params.yaml` file to point to your data and specify the location for saving the model.
3. Run the script:

```bash
python train.py
```

This will start the training process, perform hyperparameter tuning, and log the results to MLFlow. You will also see the model's accuracy, confusion matrix, and classification report printed in the terminal.

---

## **Model Evaluation**

After training, you can evaluate the model using metrics such as accuracy, confusion matrix, and classification report. The results will be available in the MLFlow UI (accessible via the tracking server URI).

- **Accuracy**: This is the primary metric used to evaluate the model performance.
- **Confusion Matrix**: This gives an insight into the model's ability to predict both classes.
- **Classification Report**: Provides precision, recall, F1 score, and support for each class.

---

## **Future Work**

- **Model Deployment**: Once the model is registered, it can be deployed as an API using frameworks such as Flask, FastAPI, or even MLFlow's model serving functionality.
- **Automation**: Automate the training pipeline with CI/CD tools like GitHub Actions or Jenkins for continuous model retraining.
- **Model Monitoring**: Implement model monitoring to track performance degradation over time and automatically trigger retraining.

---

## **Conclusion**

This project demonstrates the power of MLFlow for managing machine learning models in an MLOps pipeline. By integrating hyperparameter tuning, model evaluation, and model tracking, we can ensure that machine learning models are reproducible, manageable, and scalable in production environments.

---

### **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
