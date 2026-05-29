import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import os
import argparse

def main(data_path):
    print(f"Loading preprocessed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    X = df.drop(columns=['MedHouseVal'])
    y = df['MedHouseVal']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="mlproject_run"):
        model = RandomForestRegressor(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")
        print(f"R2: {r2}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="california_housing_preprocessing/california_housing.csv")
    args = parser.parse_args()
    main(args.data_path)
