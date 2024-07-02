import json
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

app = FastAPI()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-taxi-experiment")

class InputData(BaseModel):
    Airline: str
    Source: str
    Destination: str
    Total_Stops: int
    Date: int
    Month: int
    Year: int
    Dep_hours: int
    Dep_min: int
    Arrival_hours: int
    Arrival_min: int
    Duration_hours: int
    Duration_min: int

def get_latest_run_id(experiment_name):
    client = MlflowClient()
    experiment_id = '1'
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        experiment_id = experiment.experiment_id
        runs = client.search_runs(experiment_id, order_by=["start_time DESC"], max_results=1)
        if runs:
            return runs[0].info.run_id
    return None
def load_model_and_preprocessor():
    experiment_name = 'nyc-taxi-experiment'
    run_id = get_latest_run_id(experiment_name)
    print("the run_id is ", run_id)

    if run_id is None:
        raise Exception(f"No runs found for experiment '{experiment_name}'")

    # Load the model from MLflow using the latest run ID
    logged_model = f'runs:/{run_id}/model'
    preprocessor_uri = f"runs:/{run_id}/preprocessor"
    model = mlflow.sklearn.load_model(logged_model)
    preprocessor = mlflow.sklearn.load_model(preprocessor_uri)

    return model,preprocessor

model,preprocessor = load_model_and_preprocessor()


# # Define the preprocessing pipeline
# categorical_features = ['Airline', 'Source', 'Destination']
# encoder = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
#     ],
#     remainder='passthrough'
# )
# scaler = StandardScaler()

# preprocessor = Pipeline(steps=[
#     ('encoder', encoder),
#     ('scaler', scaler)
# ])

# Dummy DataFrame to fit the ColumnTransformer
# Adjust this based on your actual data for fitting purposes
# dummy_data = pd.DataFrame({
#     'Airline': ['Dummy_Airline'],
#     'Source': ['Dummy_Source'],
#     'Destination': ['Dummy_Destination']
# })

# # Fit the preprocessor pipeline with dummy data
# try:
#     preprocessor.fit(dummy_data)
# except Exception as e:
#     raise Exception(f"Error in fitting preprocessing pipeline: {str(e)}")

@app.post("/predict/")
def predict(data: InputData):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Preprocess the input data
    try:
        input_processed = preprocessor.transform(input_df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in preprocessing: {str(e)}")
    
    # Make a prediction
    try:
        prediction = model.predict(input_processed)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in prediction: {str(e)}")
    
    return {"predicted_price": prediction[0]}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=5001)
