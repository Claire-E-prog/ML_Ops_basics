import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow 
import mlflow.sklearn
from mlflow.models.signature import infer_signature


data = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
X = data.drop(columns=["species"])
y = data["species"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# Log the mode and metrics to MLflow
mlflow.set_experiment("Iris Classification Comparison of Models")
input_example = {"sepal_length": 5.1
                 , "sepal_width": 3.5
                 , "petal-length": 1.4
                 , "petal_width": 0.2}
# signature = infer_signature(X_train, model.predict(X_train))

def train_and_log_models(model, model_name):
    with mlflow.start_run():
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")

    