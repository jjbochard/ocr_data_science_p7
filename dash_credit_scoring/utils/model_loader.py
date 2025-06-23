import os

import mlflow
import pandas as pd
import shap
from dotenv import load_dotenv

load_dotenv(override=True)

TRACKING_URI = os.getenv("TRACKING_URI")
FINAL_RUN = os.getenv("FINAL_RUN")

mlflow.set_tracking_uri(TRACKING_URI)


def load_model_and_data():
    df = pd.read_csv(
        "data/home_credit_selected_features.csv.gz", compression="gzip"
    )
    # TODO: Remove this line when app in production
    # Filter only the first 10 rows for performance
    df = df.iloc[:10].copy()

    # Load model
    model_uri = f"runs:/{FINAL_RUN}/final_model"
    pipeline = mlflow.sklearn.load_model(model_uri)

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["processor"]

    df_no_id = df.drop(columns=["SK_ID_CURR"])
    X_transformed = preprocessor.transform(df_no_id)

    # Clean column names (last part after __)
    X_df = pd.DataFrame(X_transformed)
    X_df.columns = X_df.columns.map(lambda col: col.split("__")[-1])

    # Compute SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_df)

    # Predict probabilities
    proba = model.predict_proba(X_df)[:, 1]

    return shap_values, proba, df, X_df, explainer
