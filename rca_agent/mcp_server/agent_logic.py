import boto3
import pandas as pd
import numpy as np
import io
import json
import pickle
import shap
from sklearn.ensemble import IsolationForest

# -------------------------------------------------
# Configuration
# -------------------------------------------------
RCA_BUCKET = "ag-agent-mesh"
RCA_FOLDER = "RCA_Agent_Output"
STATUS_FOLDER = "status"

# -------------------------------------------------
# AWS Client
# -------------------------------------------------
s3 = boto3.client("s3")


# -------------------------------------------------
# Write Status File to S3
# -------------------------------------------------
def write_status(batch_id: str, status: str, message: str = None, output_path: str = None):
    """Write an agent status JSON file to S3 for tracking."""
    status_key = f"{STATUS_FOLDER}/{batch_id}.json"

    body = {
        "batch_id": batch_id,
        "status": status,
        "message": message,
        "output_path": output_path
    }

    s3.put_object(
        Bucket=RCA_BUCKET,
        Key=status_key,
        Body=json.dumps(body),
        ContentType="application/json"
    )


# -------------------------------------------------
# Core RCA Logic
# -------------------------------------------------
def run_rca(
    dataset_bucket: str,
    dataset_key: str,
    model_bucket: str,
    model_key: str,
    deviation_bucket: str,
    deviation_key: str
) -> dict:
    """
    Run Root Cause Analysis using SHAP explainability on anomalous rows.

    Args:
        dataset_bucket: S3 bucket containing the input dataset CSV
        dataset_key: S3 key of the input dataset CSV
        model_bucket: S3 bucket containing the Isolation Forest model artifact
        model_key: S3 key of the model pickle file
        deviation_bucket: S3 bucket containing the Deviation Agent output
        deviation_key: S3 key of the deviation output JSON

    Returns:
        Dictionary with batch_id of the processed batch
    """
    # ===============================
    # LOAD DATASET FROM S3
    # ===============================
    obj = s3.get_object(Bucket=dataset_bucket, Key=dataset_key)
    df = pd.read_csv(io.BytesIO(obj["Body"].read()), header=None)
    df.columns = df.columns.astype(int)

    # ===============================
    # LOAD MODEL ARTIFACT
    # ===============================
    obj = s3.get_object(Bucket=model_bucket, Key=model_key)
    artifact = pickle.loads(obj["Body"].read())

    iso_model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    df = df[feature_names]

    # ===============================
    # LOAD DEVIATION AGENT OUTPUT
    # ===============================
    obj = s3.get_object(Bucket=deviation_bucket, Key=deviation_key)
    deviation_results = json.loads(obj["Body"].read())

    anomalous_rows = [item["row_index"] for item in deviation_results]

    # ===============================
    # SPLIT DATA (Normal / Anomaly)
    # ===============================
    sensor_df = df.copy()
    anomaly_df = sensor_df.loc[anomalous_rows]

    # ===============================
    # COMPUTE ANOMALY-ONLY STATISTICS
    # ===============================
    anomaly_mean = anomaly_df.mean()
    anomaly_std = anomaly_df.std()

    # ===============================
    # SHAP EXPLAINER
    # ===============================
    explainer = shap.TreeExplainer(iso_model)

    TOP_K = 5
    rca_results = []

    def categorize_anomaly_z(z):
        if abs(z) >= 2:
            return "Large deviation among anomalies"
        elif abs(z) >= 1:
            return "Moderate deviation among anomalies"
        else:
            return "Within anomaly variation"

    for item in deviation_results:

        row_id = item["row_index"]
        anomaly_score = item["anomaly_score"]

        row_values = df.loc[row_id, feature_names]
        X_row_scaled = scaler.transform([row_values])

        shap_values = explainer.shap_values(X_row_scaled)[0]
        shap_series = pd.Series(shap_values, index=feature_names)
        top_sensors = shap_series.abs().nlargest(TOP_K)

        causes = []

        for sensor in top_sensors.index:

            value = row_values[sensor]
            a_mean = anomaly_mean[sensor]
            a_std = anomaly_std[sensor]

            z_score = 0 if a_std == 0 else (value - a_mean) / a_std
            z_category = categorize_anomaly_z(z_score)

            causes.append({
                "sensor": int(sensor),
                "shap_contribution": round(float(shap_series[sensor]), 4),
                "z_score_among_anomalies": round(float(z_score), 3),
                "z_category": z_category
            })

        rca_results.append({
            "batch_id": item["batch_id"],
            "row_index": int(row_id),
            "anomaly_score": float(anomaly_score),
            "root_causes": causes
        })

    rca_output = {
        "batch_id": deviation_results[0]["batch_id"],
        "rca_results": rca_results
    }

    rca_key = f"{RCA_FOLDER}/{deviation_results[0]['batch_id']}.json"

    s3.put_object(
        Bucket=RCA_BUCKET,
        Key=rca_key,
        Body=json.dumps(rca_output),
        ContentType="application/json"
    )

    return {
        "batch_id": deviation_results[0]["batch_id"]
    }
