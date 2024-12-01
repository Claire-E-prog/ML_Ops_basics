import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

# Dummy data
np.random.seed(123)
X = np.random.rand(100,1)
y = 3 * X.squeeze() + 2 + np.random.randn(100) * 5

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# Experiments

mlflow.set_experiment("Comparison of Models")

def train_log_model(model, model_name):
    with mlflow.start_run(run_name = model_name): # Strats tracking
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # mlflow.log_params(model.get_params())
        mlflow.log_param("fit intercept", model.fit_intercept)
        
        # mlflow.log_metrics({"mse": mse, "r2": r2})
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        print(f"Model {model_name} trained and loggged")

train_log_model(LinearRegression(), "Linear Regression")


